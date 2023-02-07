import tokenizers
import torch

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.init()
vocab_size = 30522
min_frequency = 3
limit_alphabet = 1000

bwpt = tokenizers.BertWordPieceTokenizer(
    unk_token = '[UNK]',
    sep_token = '[SEP]',
    cls_token = '[CLS]',
    clean_text = True,
    handle_chinese_chars = True,
    strip_accents = False,
    lowercase = False,
    wordpieces_prefix = '##',
)

bwpt.train(
    files = ['/home/fm-pc-lt-228/Desktop/resume_job_dump/technicalBERT/train.txt'],
    vocab_size = vocab_size,
    min_frequency = min_frequency,
    limit_alphabet = limit_alphabet,
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]'],
)

bwpt.save_model('bert_custom_tokenizer/')