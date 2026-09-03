"""Minimal robust reader for the Blender binary PLY files used by paper LND."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": ("b", np.dtype("i1")),
    "int8": ("b", np.dtype("i1")),
    "uchar": ("B", np.dtype("u1")),
    "uint8": ("B", np.dtype("u1")),
    "short": ("h", np.dtype("<i2")),
    "int16": ("h", np.dtype("<i2")),
    "ushort": ("H", np.dtype("<u2")),
    "uint16": ("H", np.dtype("<u2")),
    "int": ("i", np.dtype("<i4")),
    "int32": ("i", np.dtype("<i4")),
    "uint": ("I", np.dtype("<u4")),
    "uint32": ("I", np.dtype("<u4")),
    "float": ("f", np.dtype("<f4")),
    "float32": ("f", np.dtype("<f4")),
    "double": ("d", np.dtype("<f8")),
    "float64": ("d", np.dtype("<f8")),
}


def load_binary_ply_triangles(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and triangulate all valid polygon faces.

    Several upstream Blender PLYs include edge elements and non-triangle
    polygons.  The commonly installed trimesh/Open3D readers either reject
    those files or return a partially decoded mesh, while the paper renderer
    accepts them.  This reader follows the declared element/property stream
    and fan-triangulates polygons without changing vertex coordinates.
    """

    with path.open("rb") as handle:
        first = handle.readline().decode("ascii").strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        format_line = handle.readline().decode("ascii").strip()
        if format_line != "format binary_little_endian 1.0":
            raise ValueError(
                f"Expected binary_little_endian PLY in {path}, got "
                f"{format_line!r}"
            )
        elements: list[dict] = []
        current: dict | None = None
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise RuntimeError(f"Missing end_header in {path}")
            line = line_bytes.decode("ascii").strip()
            if line == "end_header":
                break
            fields = line.split()
            if not fields or fields[0] in {"comment", "obj_info"}:
                continue
            if fields[0] == "element":
                current = {
                    "name": fields[1],
                    "count": int(fields[2]),
                    "properties": [],
                }
                elements.append(current)
            elif fields[0] == "property":
                if current is None:
                    raise RuntimeError(f"PLY property before element in {path}")
                if fields[1] == "list":
                    current["properties"].append(
                        ("list", fields[2], fields[3], fields[4])
                    )
                else:
                    current["properties"].append(
                        ("scalar", fields[1], fields[2])
                    )
            else:
                raise ValueError(f"Unsupported PLY header line: {line}")

        vertices: np.ndarray | None = None
        triangles: list[tuple[int, int, int]] = []
        for element in elements:
            name = str(element["name"])
            count = int(element["count"])
            properties = element["properties"]
            scalar_only = all(item[0] == "scalar" for item in properties)
            if scalar_only:
                dtype = np.dtype(
                    [
                        (item[2], PLY_TYPES[item[1]][1])
                        for item in properties
                    ]
                )
                raw = handle.read(count * dtype.itemsize)
                if len(raw) != count * dtype.itemsize:
                    raise RuntimeError(
                        f"Truncated {name} element in {path}"
                    )
                values = np.frombuffer(raw, dtype=dtype, count=count)
                if name == "vertex":
                    required = {"x", "y", "z"}
                    if not required.issubset(values.dtype.names or ()):
                        raise ValueError(f"PLY vertices lack xyz in {path}")
                    vertices = np.column_stack(
                        [values["x"], values["y"], values["z"]]
                    ).astype(np.float32)
                continue

            for _ in range(count):
                face_indices: list[int] | None = None
                for prop in properties:
                    if prop[0] == "scalar":
                        format_char = PLY_TYPES[prop[1]][0]
                        size = struct.calcsize("<" + format_char)
                        raw = handle.read(size)
                        if len(raw) != size:
                            raise RuntimeError(
                                f"Truncated scalar PLY property in {path}"
                            )
                    else:
                        count_format = PLY_TYPES[prop[1]][0]
                        item_format = PLY_TYPES[prop[2]][0]
                        count_size = struct.calcsize("<" + count_format)
                        raw_count = handle.read(count_size)
                        if len(raw_count) != count_size:
                            raise RuntimeError(
                                f"Truncated list count in {path}"
                            )
                        list_count = int(
                            struct.unpack("<" + count_format, raw_count)[0]
                        )
                        item_size = struct.calcsize("<" + item_format)
                        raw_items = handle.read(list_count * item_size)
                        if len(raw_items) != list_count * item_size:
                            raise RuntimeError(
                                f"Truncated list property in {path}"
                            )
                        items = list(
                            struct.unpack(
                                "<" + item_format * list_count,
                                raw_items,
                            )
                        )
                        if (
                            name == "face"
                            and prop[3] in {"vertex_indices", "vertex_index"}
                        ):
                            face_indices = [int(item) for item in items]
                if name == "face" and face_indices is not None:
                    for index in range(1, len(face_indices) - 1):
                        triangles.append(
                            (
                                face_indices[0],
                                face_indices[index],
                                face_indices[index + 1],
                            )
                        )

        trailing = handle.read()
        if trailing and any(byte != 0 for byte in trailing):
            raise RuntimeError(
                f"PLY stream has {len(trailing)} unexplained trailing bytes: "
                f"{path}"
            )
    if vertices is None:
        raise RuntimeError(f"PLY contains no vertex element: {path}")
    faces = np.asarray(triangles, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise RuntimeError(f"PLY contains no usable faces: {path}")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError(f"PLY face index is out of range: {path}")
    return vertices, faces
