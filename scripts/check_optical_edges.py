from nugraph.data import H5DataModule

graph_file = "merged_large.graph.h5.0000.h5"

dm = H5DataModule(
    data_path=graph_file,
    batch_size=16,
    num_workers=0,
)

def check_loader(name, loader):
    print(f"\nChecking {name}")
    bad = 0

    for ib, batch in enumerate(loader):
        for edge_type in batch.edge_types:
            src, rel, dst = edge_type
            ei = batch[edge_type].edge_index

            if ei.numel() == 0:
                continue

            src_n = batch[src].num_nodes
            dst_n = batch[dst].num_nodes

            src_min = int(ei[0].min())
            src_max = int(ei[0].max())
            dst_min = int(ei[1].min())
            dst_max = int(ei[1].max())

            ok = (
                src_min >= 0 and
                dst_min >= 0 and
                src_max < src_n and
                dst_max < dst_n
            )

            if not ok:
                bad += 1
                print("\nBAD EDGE FOUND")
                print("split =", name)
                print("batch =", ib)
                print("edge_type =", edge_type)
                print("edge_index shape =", tuple(ei.shape))
                print("src nodes:", src, src_n, "min/max:", src_min, src_max)
                print("dst nodes:", dst, dst_n, "min/max:", dst_min, dst_max)
                print("metadata run =", batch.metadata["run"])
                print("metadata subrun =", batch.metadata["subrun"])
                print("metadata event =", batch.metadata["event"])
                return False

    print("OK:", name)
    return True

dm.setup("fit")
ok_train = check_loader("train", dm.train_dataloader())
ok_val = check_loader("validation", dm.val_dataloader())

dm.setup("test")
ok_test = check_loader("test", dm.test_dataloader())

print("\nSUMMARY")
print("train:", ok_train)
print("validation:", ok_val)
print("test:", ok_test)
