from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "davidkim205/Ko-Llama-3-8B-Instruct"

class llama:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def msg(self,prompt):
        while True:
            messages = [
                {"role": "system", "content": "빠르고 간결하게 답변을 하는 챗봇입니다."},
                {"role": "user", "content": prompt},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            # print(self.tokenizer.decode(response, skip_special_tokens=True))
            return self.tokenizer.decode(response, skip_special_tokens=True)