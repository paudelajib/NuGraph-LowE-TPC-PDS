from pathlib import Path

path = Path("/home/apaudel/NuGraph/pynuml/pynuml/process/hitgraph.py")
lines = path.read_text().splitlines(keepends=True)

start = None
end = None

for i, line in enumerate(lines):
    if line.strip() == "# spacepoint -> nearest PMT edges":
        start = i
        break

if start is None:
    raise RuntimeError("Could not find KNN start marker")

for i in range(start + 1, len(lines)):
    if lines[i].strip() == "# event label":
        end = i
        break

if end is None:
    raise RuntimeError("Could not find event label marker")

indent = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
i1 = indent
i2 = indent + "    "
i3 = indent + "        "
i4 = indent + "            "

new = [
    f"{i1}# spacepoint -> nearest PMT edges\n",
    f'{i1}if spacepoints_nodes.shape[0] > 0 and data["pmt"].pos.shape[0] > 0:\n',
    f"{i2}distances = torch.cdist(spacepoints_nodes.float(), data[\"pmt\"].pos.float())\n",
    f"{i2}k = min(int(nnear), int(distances.shape[1]))\n",
    f"{i2}if k > 0:\n",
    f"{i3}_, nearest_indices = torch.topk(distances, k, largest=False, dim=1)\n",
    f"{i3}spacepoints_indices = torch.arange(spacepoints_nodes.shape[0], dtype=torch.long).repeat_interleave(k)\n",
    f"{i3}opflashsumpe_indices = nearest_indices.reshape(-1).long()\n",
    f'{i3}data["sp", "knn", "pmt"].edge_index = torch.stack([spacepoints_indices, opflashsumpe_indices], dim=0).long()\n',
    f"{i2}else:\n",
    f'{i3}data["sp", "knn", "pmt"].edge_index = torch.empty((2, 0), dtype=torch.long)\n',
    f"{i1}else:\n",
    f'{i2}data["sp", "knn", "pmt"].edge_index = torch.empty((2, 0), dtype=torch.long)\n',
    "\n",
]

path.write_text("".join(lines[:start] + new + lines[end:]))
print(f"Patched lines {start+1} to {end} in {path}")
