from datasets import load_dataset
import json
import os


def main():
    if not os.path.exists("data"):
        os.mkdir("data")

    print("Fetching the training data from s3://forensic-training-data/jsonl/")
    os.system(f"aws s3 cp s3://forensic-training-data/jsonl/ data/ --recursive")
    

    with open("data/tokens.json", "w") as f:
        tokens = {}
        tokens["tokens"] = ["[INST]", "<<SYS>>", "<</SYS>>", "[/INST]"]
        print('Adding special tokens to the vocabulary: ' + str(tokens["tokens"]))
        f.write(json.dumps(tokens))

    print('[ACTION REQUIRED] Copy the training data from verita to ./data')


if __name__ == "__main__":
    main()
