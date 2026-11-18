// Axes are packed innermost first. Contiguous or broadcast runs are collapsed
// on the host. Metal binds the tensor's storage offset at the buffer interface.
inline ulong2 pointwise_offsets(uint i, constant long* layout) {
    ulong2 offsets = 0;
    uint rank = uint(layout[0]);
    for (uint d=0; d<rank; ++d) {
        uint coordinate = i;
        if (d+1 < rank) {
            uint width = uint(layout[1+3*d]);
            uint outer = i/width;
            coordinate = i-outer*width;
            i = outer;
        }
        offsets += ulong(coordinate) * ulong2(layout[2+3*d], layout[3+3*d]);
    }
    return offsets;
}
