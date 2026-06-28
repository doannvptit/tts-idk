from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def train_bbpe_from_hf(vocab_size=3072):
    print("Đang tải Wikipedia tiếng Việt (bản an toàn Parquet) từ Hugging Face...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train")
    max_samples = min(25000, len(dataset))
    print(f"Tổng số bài viết hiện có: {len(dataset)}. Sẽ dùng {max_samples} bài viết để huấn luyện.")

    def batch_iterator(batch_size=500):
        for i in range(0, max_samples, batch_size):
            sub_set = dataset.select(range(i, min(i + batch_size, max_samples)))
            for text in sub_set["text"]:
                if text and len(text.strip()) > 0: 
                    yield text

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    
    special_tokens = [
        "[PAD]", "<AUDIO>", "</AUDIO>", "<BOS>", "<EOS>", 
        "<VOICE_CLONE>", "</VOICE_CLONE>"
    ] + [f"[audio_token_{i}]" for i in range(1024)]
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet()
    )
    
    print(f"Bắt đầu huấn luyện BBPE với vocab_size = {vocab_size}...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    model_name = f"vi_wikipedia_bbpe_{vocab_size}.json"
    tokenizer.save(model_name)
    print(f"🎉 Huấn luyện thành công! File lưu tại: {model_name}")
    
    return tokenizer

# === Chạy kiểm thử ===
if __name__ == "__main__":
    vi_tokenizer = train_bbpe_from_hf(vocab_size=2048)
    
    test_sentence = "Công nghệ thông tin và trí tuệ nhân tạo đang thay đổi Việt Nam."
    output = vi_tokenizer.encode(test_sentence)
    
    print("\n--- KẾT QUẢ IN TOKEN ---")
    print(f"Câu gốc: {test_sentence}")
    print(f"Tokens ID: {output.ids}")
    print(f"Tokens chữ: {output.tokens}")
    print(f"Giải mã (Decode): {vi_tokenizer.decode(output.ids)}")
    
    # Kiểm tra thử nghiệm token đặc biệt
    test_special = "<BOS> <AUDIO> Xin chào Việt Nam </AUDIO> <EOS>"
    output_special = vi_tokenizer.encode(test_special, add_special_tokens=False)
    print(f"\nTest token đặc biệt: {output_special.tokens}")