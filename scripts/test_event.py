#!/usr/bin/env python

import os
import time
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
import nugraph as ng
import tqdm

Data = ng.data.H5DataModule
Model = ng.models.NuGraph3


def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    parser = Data.add_data_args(parser)
    return parser.parse_args()


def test(args):
    print("data path =", args.data_path)

    nudata = Data(
        args.data_path,
        batch_size=args.batch_size,
        model=Model,
        num_workers=args.num_workers,
    )
    nudata.setup("test")

    print("using checkpoint =", args.checkpoint)
    model = Model.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()

    print("output file =", args.outfile)
    if os.path.isfile(args.outfile):
        raise Exception(f"file {args.outfile} already exists!")

    accelerator, devices = ng.util.configure_device(args.device)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
    )

    start = time.time()
    out = trainer.predict(model, dataloaders=nudata.test_dataloader())
    end = time.time()

    ngraphs = len(nudata.test_dataset)
    print(f"inference for {ngraphs} events is {end-start:.2f} s")

    rows = []
    classes = list(nudata.event_classes)
    print("event classes =", classes)

    for batch in tqdm.tqdm(out):
        for data in batch.to_data_list():
            if not hasattr(data["evt"], "e"):
                raise RuntimeError(
                    "No event prediction found: data['evt'].e is missing. "
                    "This checkpoint may not have been trained with --event."
                )

            y_true = int(data["evt"].y.item())
            probs = data["evt"].e.detach().cpu().reshape(-1)
            y_pred = int(torch.argmax(probs).item())

            row = {
                "run": int(data["metadata"].run),
                "subrun": int(data["metadata"].subrun),
                "event": int(data["metadata"].event),
                "y_true": y_true,
                "y_pred": y_pred,
            }

            for i, cname in enumerate(classes):
                row[f"prob_{cname}"] = float(probs[i].item())

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_hdf(args.outfile, key="events", format="table")

    print("saved:", args.outfile)
    print(df.head())
    print("")
    print("counts true:")
    print(df["y_true"].value_counts().sort_index())
    print("")
    print("counts pred:")
    print(df["y_pred"].value_counts().sort_index())


if __name__ == "__main__":
    args = configure()
    test(args)
