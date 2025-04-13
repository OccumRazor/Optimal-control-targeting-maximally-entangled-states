space_symbol = ' '
space4_symbol = '    '
space5_symbol = '     '

def pulse_title(is_complex):
    text_info = '#'
    text_info += space_symbol * 18 + 'time'
    text_info += space4_symbol
    text_info += space_symbol * 19 + 'real'
    if is_complex == True:text_info += space_symbol * (19 + 4) + 'imag'
    return text_info + '\n'

def state_title(state_shape,is_complex):
    len_max_row = len(str(state_shape[0]))
    text_info = f'# n_row: {state_shape[0]}\n#'
    text_info += space_symbol * (len_max_row + 2 - 3) + 'col'
    text_info += space5_symbol
    text_info += space_symbol * 18 + 'real'
    if is_complex:text_info += space_symbol * (18 + 5) + 'imag'
    return text_info + '\n'

def matrix_title(matrix_shape,is_complex):
    len_max_row = len(str(matrix_shape[0]))
    text_info = f'# n_row: {matrix_shape[0]} n_col: {matrix_shape[1]}\n#'
    text_info += space_symbol * (len_max_row + 2 - 3) + 'row'
    text_info += space_symbol * len_max_row + 'col'
    text_info += space5_symbol
    text_info += space_symbol * 18 + 'real'
    if is_complex:text_info += space_symbol * (18 + 5) + 'imag'
    return text_info + '\n'