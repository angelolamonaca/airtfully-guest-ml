from diffusers import DiffusionPipeline


def main():
    pipeline = DiffusionPipeline.from_pretrained("../openjourney-v4")
    pipeline.to("cuda")
    pipeline.enable_attention_slicing("max")
    img = pipeline("An image of a squirrel in Picasso style").images[0]
    img.save("output/geeks.jpg")


if __name__ == '__main__':
    main()
