import argparse
import torch
from transformers import AutoModel
from safetensors.torch import save_file


def convert_bin_to_safetensors(input_bin_path, output_safetensors_path, model_name_or_path):
    # Load the model from the bin file
    model = AutoModel.from_pretrained(model_name_or_path, state_dict=torch.load(input_bin_path))

    # Get the model's state dictionary
    state_dict = model.state_dict()

    # Save the state dictionary to safetensors format
    save_file(state_dict, output_safetensors_path)
    print(f"Converted {input_bin_path} to {output_safetensors_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch .bin weights to safetensors format")
    parser.add_argument("--bin_path", type=str, required=True, help="Path to the input .bin file")
    parser.add_argument("--sf_path", type=str, required=True,
                        help="Path to the output .safetensors file")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Name or path of the model")

    args = parser.parse_args()

    convert_bin_to_safetensors(args.input_bin_path, args.output_safetensors_path, args.model_name_or_path)


if __name__ == "__main__":
    main()
