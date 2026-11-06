SYSTEM = 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
MLDR_INSTRUCTION = DEFAULT_INSTRUCTION
PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"
THINK = "<think>\n\n</think>\n\n"
SUFFIX = ASSISTANT + THINK


def format_content(query, document, instruction, separator="\n\n"):
    return separator.join((f"<Instruct>: {instruction}", f"<Query>: {query}", f"<Document>: {document}"))


def token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected one token for {text!r}")
    return ids[0]


def encode_pair(tokenizer, query, document, instruction, max_length=8192):
    if max_length < 1:
        raise ValueError("max_length must be positive")
    content = format_content(query, document, instruction)
    body = tokenizer.encode(PREFIX + content, add_special_tokens=False)[:max_length]
    return body + tokenizer.encode(SUFFIX, add_special_tokens=False)


def encode_training_pair(tokenizer, query, document, instruction, positive, max_length=4096):
    source = PREFIX + format_content(query, document, instruction, "\n") + ASSISTANT
    target = THINK + ("yes" if positive else "no") + "<|im_end|>\n"
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if max_length <= 2 * len(target_ids):
        raise ValueError("max_length is too small for the training response")
    source_ids = source_ids[:max_length - len(target_ids)]
    labels = [-100] * len(source_ids) + target_ids
    yes, no = token_id(tokenizer, "yes"), token_id(tokenizer, "no")
    positions = [i for i, value in enumerate(labels) if value in (yes, no)]
    if len(positions) != 1 or positions[0] < 1:
        raise ValueError("Training response must contain one yes/no token")
    return {"input_ids": source_ids + target_ids, "labels": labels, "score_position": positions[0] - 1}
