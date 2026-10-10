# Brightness depends on distance but color doesn't, so colors are better.
# The five filter values are magnitudes, which work backwards: a smaller number means brighter.
# The difference between two filters (e.g. u − g) is called a color and tells you whether
# the object gives off more blue or more red light.


def add_colors(df):
    """Add the four adjacent-filter colors as new columns."""
    df = df.copy()
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    return df
