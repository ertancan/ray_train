from datasets import load_dataset
import json
import os


def main():
    if not os.path.exists("data"):
        os.mkdir("data")

    print("Fetching the training data from s3://forensic-training-data/ray/")
    os.system(f"aws s3 cp s3://forensic-training-data/ray/ training_data/ --recursive")
    

    with open("data/tokens.json", "w") as f:
        tokens = {}
        tokens["tokens"] = ["[INST]", "<<SYS>>", "<</SYS>>", "[/INST]", "<START_A>"]
        print('Adding special tokens to the vocabulary: ' + str(tokens["tokens"]))
        f.write(json.dumps(tokens))


if __name__ == "__main__":
    main()
