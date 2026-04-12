from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

from time import perf_counter


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]: #返回 当前step要执行的一批seqs和is_prefill
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs: # 限制了一个 batch 中最多可以处理多少个 seqs
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break # Token 数量限制
                #由于 Prefill 阶段通常需要一次性计算 整个 prompt 的 token ，
                # 如果 batch 中 token 总数过多，就可能导致显存不足、GPU 计算压力过大，
                # 因此 Scheduler 需要限制 一个 batch 的最大 token 数量。
            num_seqs += 1
            self.block_manager.allocate(seq) #blck manager 用来管理kv cache的内存分配
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq) #移动到running队列
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs: #当前没有新的prefill请求需要执行
            seq = self.running.popleft()
            # 每个 sequence 只生成一个 token，因此 decode 的计算量相对较小，
            # 但 需要频繁执行，在这个过程中 Scheduler 仍然需要检查 KV Cache 是否能够继续扩展
            # 因为每生成一个新的 token，都需要分配新的 KV Cache。
            # 如果 KV Cache 空间不足，就会触发一个非常重要的机制 Preemption（抢占）
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop()) #抢占一个 running sequence，释放资源
                else:
                    self.preempt(seq) #抢占当前seq自己
                    break
            else:
                num_seqs += 1 #每生成一个新token seq长度增加一位
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq) #并非放回末尾，而是放回优先位置，因为这个seq已经执行过一部分了，不是一个全新请求

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        finished = []
        now = perf_counter()
        for seq, token_id in zip(seqs, token_ids):
            if seq.first_token_time is None:
                seq.first_token_time = now
                if seq.arrival_time is not None:
                    seq.ttft = seq.first_token_time - seq.arrival_time

            seq.append_token(token_id)

            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                finished.append((seq.seq_id, seq.completion_token_ids, seq.ttft))
        return finished
'''    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) #追加生成token
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED #标记完成
                self.block_manager.deallocate(seq) #释放kv cache
                self.running.remove(seq) #从running队列移除
                '''
