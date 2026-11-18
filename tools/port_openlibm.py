"""Generate Metal source from selected, pinned OpenLibm C sources using clang -E."""

import pathlib
import re
import struct
import subprocess
import sys

src = pathlib.Path(sys.argv[1]) / "src"
out = pathlib.Path(__file__).resolve().parents[1] / "mps_fp64"
files = "s_floor s_scalbn e_exp e_log s_expm1 s_log1p s_atan e_atan2 e_asin e_acos s_tanh k_sin k_cos k_tan k_rem_pio2 e_rem_pio2 s_sin s_cos s_tan e_pow s_ceil s_trunc s_erf s_asinh e_acosh e_atanh e_hypot k_exp e_sinh e_cosh e_log2 e_log10 s_exp2 e_fmod".split()
symbols = {}
for name in files:
    category, base = name.split("_", 1)
    symbols[
        {"s": base, "e": "__ieee754_" + base, "k": "__kernel_" + base}[category]
    ] = name
floating = re.compile(
    r"\b(?:0[xX][0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?[pP][+-]?\d+|\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)\b"
)


def literal(match):
    s = match[0]
    value = float.fromhex(s) if s.lower().startswith("0x") else float(s)
    bits = int.from_bytes(struct.pack(">d", value), "big")
    return f"Soft(0x{bits:016x}ul,true)"


pieces = []
for name in files:
    source = (src / (name + ".c")).read_text()
    if name in ("e_log2", "e_log10"):
        source = source.replace('#include "k_log.h"', (src / "k_log.h").read_text())
    if name == "k_exp":
        source = source.split("OLM_DLLEXPORT double complex")[0]
    notice = source[: source.index("#include")]
    source = re.sub(r"^\s*#include[^\n]*", "", source, flags=re.M)
    source = subprocess.run(
        ["clang", "-E", "-P", "-x", "c", "-"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    source = re.sub(
        r"\b(static\s+(?:const|volatile)|static\s+const\s+volatile)\b",
        "constant",
        source,
    )
    source = re.sub(r"\b(double|__double_t)\b", "Soft", source)
    source = source.replace("OLM_DLLEXPORT", "inline").replace("u_int32_t", "uint")
    source = source.replace("Soft *", "thread Soft *")
    source = re.sub(r"openlibm_strong_reference\([^;]+;", "", source)
    source = re.sub(r"\bhalf\b", "one_half", source)
    source = source.replace("*(volatile thread Soft *)&atanlo[3]", "atanlo[3]")
    source = re.sub(r"k = (invln2\*x\+[^;]+);", r"k = int(\1);", source)
    if name == "k_rem_pio2":
        source = source.replace("recompute:", "while (true) {").replace(
            "goto recompute;", "continue;"
        )
        source = source.replace("if(z==0.0)", "break; }\n if(z==0.0)")
    if name == "e_rem_pio2":
        source = source.replace("if (ix <= 0x400f6a7a)", "do { if (ix <= 0x400f6a7a)")
        source = (
            source.replace("goto medium;", "break;")
            .replace("if(ix<0x413921fb)", "} while(false); if(ix<0x413921fb)")
            .replace("medium:", "")
        )
    if name == "k_exp":
        source = source.replace("int *expt", "thread int *expt")
    for dep in ("exp", "log"):
        if name != "e_" + dep:
            source = re.sub(
                r"\b" + dep + r"\(", "ol_e_" + dep + "::__ieee754_" + dep + "(", source
            )
    if name in ("e_sinh", "e_cosh"):
        source = source.replace("__ldexp_exp(", "ol_k_exp::__ldexp_exp(")
    source = source.replace("__ieee754_sqrt(", "sqrt(")
    if name == "s_exp2":
        # Fold the exact power-of-two scale in the constant initializer.
        source = source.replace("0x1.8p52 / (1 << 8)", "0x1.8p44")
    source = floating.sub(literal, source)
    for symbol, owner in symbols.items():
        if owner != name:
            source = re.sub(
                r"(?<!::)\b" + symbol + r"\s*\(",
                "ol_" + owner + "::" + symbol + "(",
                source,
            )
    pieces.append(f"{notice}\nnamespace ol_{name} {{\n{source}\n}}\n")
(out / "vendor/openlibm.metal").write_text("\n".join(pieces))
