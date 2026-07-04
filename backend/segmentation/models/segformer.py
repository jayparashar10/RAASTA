from transformers import SegformerForSemanticSegmentation


def load_segformer():
    """
    Load pretrained SegFormer model.
    """

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    return model