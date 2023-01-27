# Ray-StableDiffusion

Runs Text2Image Stable Diffusion fine tuning using Ray and Accelerate.

Training is done with Ray Train `TorchTrainer`, using Accelerate inside the training function. Data preprocessing is done with Ray Datasets.

Needs Ray>=2.3 (nightly at the time of writing).

Install requirements and run `bash run.sh`. You can change the arguments there.

Same arguments as in https://huggingface.co/docs/diffusers/training/text2image + a few extra. Run `python train_text_to_image.py --help` to see the entire list.

In order to run on 16 GB or less GPUs, make sure to use [DeepSpeed](https://huggingface.co/docs/diffusers/training/dreambooth#training-on-a-8-gb-gpu) (`--use-deepspeed`) or [LoRA](https://huggingface.co/docs/diffusers/training/lora#getting-started-with-lora-for-finetuning) (`--use-lora`). Using both doesn't seem to give huge benefit.

Based on https://github.com/huggingface/diffusers/tree/main/examples/text_to_image. The regular and LoRA examples have been combined into one.
