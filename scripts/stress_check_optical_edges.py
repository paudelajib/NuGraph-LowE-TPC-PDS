from nugraph.data import H5DataModule

graph_file = "merged_large.graph.h5.0000.h5"

for repeat in range(20):
    dm = H5DataModule(
        data_path=graph_file,
        batch_size=16,
        num_workers=0,
    )
    dm.setup("fit")

    print("shuffle pass", repeat)

    for ib, batch in enumerate(dm.train_dataloader()):
        for edge_type in [
            ("ophit", "in", "pmt"),
            ("pmt", "in", "flash"),
            ("flash", "in", "evt"),
            ("sp", "knn", "pmt"),
        ]:
            src, rel, dst = edge_type
            ei = batch[edge_type].edge_index

            if ei.numel() == 0:
                continue

            src_n = batch[src].num_nodes
            dst_n = batch[dst].num_nodes

            if int(ei[0].min()) < 0 or int(ei[0].max()) >= src_n:
                print("BAD SRC")
                print("repeat", repeat, "batch", ib, edge_type)
                print("src_n", src_n, "src min/max", int(ei[0].min()), int(ei[0].max()))
                print("metadata event", batch.metadata["event"])
                raise SystemExit

            if int(ei[1].min()) < 0 or int(ei[1].max()) >= dst_n:
                print("BAD DST")
                print("repeat", repeat, "batch", ib, edge_type)
                print("dst_n", dst_n, "dst min/max", int(ei[1].min()), int(ei[1].max()))
                print("metadata event", batch.metadata["event"])
                raise SystemExit

print("ALL SHUFFLE PASSES OK")
