from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    Blocks (or tokens) layout:

    ----------------------------------------------------------------------
    | < computed > | < new_computed > |       < new >       |
    ----------------------------------------------------------------------
    |     < Prefix-cached tokens >    |  < to be computed > |
    ----------------------------------------------------------------------
                                      | < to be allocated > |
    ----------------------------------------------------------------------
                                      |   < to be cached >  |
    ----------------------------------------------------------------------
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if self.hash_to_block_id.get(block.hash) == block_id:
            self.hash_to_block_id.pop(block.hash, None)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _ensure_seq_block(self, seq: Sequence, block_index: int) -> Block:
        while len(seq.block_table) <= block_index:
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
        return self.blocks[seq.block_table[block_index]]

    def _clear_block_hash(self, block: Block):
        if self.hash_to_block_id.get(block.hash) == block.block_id:
            self.hash_to_block_id.pop(block.hash, None)
        block.hash = -1

    def can_allocate(self, num_tokens: int) -> bool:
        """
        Only for seq in the waiting queue.
        """
        return len(self.free_block_ids) >= (num_tokens + self.block_size - 1) // self.block_size

    def get_token_layout(self, seq: Sequence):
        """
        Only for seq in the waiting queue.
        """
        assert not seq.block_table
        num_new_tokens = 0
        num_new_computed_tokens_in_used = 0
        num_new_computed_tokens_in_free = 0
        h = -1
        cache_miss = False

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            # 最后一个 block 永远视为 miss，保证能重新算到序列尾部拿到 logits
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids or i == seq.num_blocks - 1:
                cache_miss = True

            if cache_miss:
                num_new_tokens += len(token_ids)
            else:
                if block_id in self.used_block_ids:
                    num_new_computed_tokens_in_used += len(token_ids)
                else:
                    num_new_computed_tokens_in_free += len(token_ids)

        return num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, num_new_tokens

    def allocate(self, seq: Sequence):
        """
        Only for seq in the waiting queue.
        """
        assert not seq.block_table
        h = -1

        # 1) 先挂 prefix-cached full blocks
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            if block_id == -1 or self.blocks[block_id].token_ids != token_ids or i == seq.num_blocks - 1:
                break

            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)

            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

        # 2) 为本轮要新算的 tokens 分配 blocks
        start = seq.num_cached_tokens
        end = seq.num_cached_tokens + seq.num_new_tokens
        while start < end:
            token_ids = seq[start:min(start + self.block_size, end)]
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)

            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            else:
                # partial block 不能 hash，但保留 token_ids 便于调试
                block.token_ids = token_ids

            seq.block_table.append(block_id)
            start += self.block_size

    def deallocate(self, seq: Sequence):
        """
        For finished seq or preempted seq in the running queue.
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.num_new_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """
        Only for seq in the running queue.
        """
        if num_new_tokens <= 0:
            return True

        tail = seq.num_cached_tokens % self.block_size
        last_computed_block_capacity = 0 if tail == 0 else (self.block_size - tail)
        remaining = max(num_new_tokens - last_computed_block_capacity, 0)
        required_new_blocks = (remaining + self.block_size - 1) // self.block_size
        return required_new_blocks <= len(self.free_block_ids)

    def may_append(self, seq: Sequence):
        """
        Only for seq in the running queue.
        把本轮新增 token 对应的 block_table 状态补齐：
        - partial block 始终保持 hash == -1
        - 只有块补满时才 finalize / 计算 hash
        """
        num_new_tokens = seq.num_new_tokens
        if num_new_tokens <= 0:
            return

        bs = self.block_size
        total_tokens = seq.num_cached_tokens + seq.num_new_tokens
        current_block_index = seq.num_cached_tokens // bs
        tail = seq.num_cached_tokens % bs

        # 1) 先处理当前 partial block（如果有）
        if tail != 0:
            current_block = self._ensure_seq_block(seq, current_block_index)

            # partial block 不能带 hash
            if current_block.hash != -1:
                self._clear_block_hash(current_block)

            current_block.token_ids = seq[current_block_index * bs: min((current_block_index + 1) * bs, total_tokens)]

            block_end = (current_block_index + 1) * bs
            if total_tokens >= block_end:
                token_ids = seq[current_block_index * bs:block_end]
                previous_block_id = seq.block_table[current_block_index - 1] if current_block_index > 0 else -1
                prefix = self.blocks[previous_block_id].hash if previous_block_id != -1 else -1
                h = self.compute_hash(token_ids, prefix)
                current_block.update(h, token_ids)
                self.hash_to_block_id[h] = current_block.block_id
                current_block_index += 1

        # 2) 处理中间所有完整的新 block
        start = current_block_index * bs
        while start + bs <= total_tokens:
            current_block = self._ensure_seq_block(seq, current_block_index)
            token_ids = seq[start:start + bs]
            previous_block_id = seq.block_table[current_block_index - 1] if current_block_index > 0 else -1
            prefix = self.blocks[previous_block_id].hash if previous_block_id != -1 else -1
            h = self.compute_hash(token_ids, prefix)
            current_block.update(h, token_ids)
            self.hash_to_block_id[h] = current_block.block_id
            current_block_index += 1
            start += bs

        # 3) 最后一个 trailing partial block
        if start < total_tokens:
            current_block = self._ensure_seq_block(seq, current_block_index)
            if current_block.hash != -1:
                self._clear_block_hash(current_block)
            current_block.token_ids = seq[start:total_tokens]
