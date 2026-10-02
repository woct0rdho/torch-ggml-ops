"""Shared validation and arithmetic for backward work-group mappings."""

_U32_MAX = 0xFFFFFFFF


def work_group_mapping_shift(mapping: int) -> int:
    """Return the shift used to decode a power-of-two mapping lane."""
    assert 0 < mapping <= _U32_MAX
    assert not (mapping & (mapping - 1))
    return mapping.bit_length() - 1


def mapped_m_tile_count(m_tiles: int, mapping: int) -> int:
    """Return the outer M-tile count after applying a work-group mapping."""
    work_group_mapping_shift(mapping)
    assert m_tiles > 0
    assert m_tiles % mapping == 0
    result = m_tiles // mapping
    assert result > 0
    return result


def mapped_grid_extent(tile_count: int, mapping: int) -> int:
    """Return a mapping-expanded grid extent that fits HIP's u32 grid ABI."""
    work_group_mapping_shift(mapping)
    assert tile_count > 0
    result = tile_count * mapping
    assert result <= _U32_MAX
    return result


def mapped_route_stride(split_factor: int, mapping: int) -> int:
    """Return the M-task stride after composing route split and WGM."""
    work_group_mapping_shift(mapping)
    assert split_factor > 0
    result = split_factor * mapping
    assert result <= _U32_MAX
    return result
