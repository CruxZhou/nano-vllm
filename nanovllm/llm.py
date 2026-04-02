from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass #直接继承LLMEngine，没有额外实现
    #完整vllm中这一层有内容
    # 提供 同步 / 异步 engine 的封装
    # 适配 OpenAI API 风格接口
    # 提供更高层的用户接口