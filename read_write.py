import numpy as np
from scipy.sparse import coo_matrix,coo_array

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

def state2text(state):
    if not isinstance(state,coo_array):state=coo_array(state)
    is_complex = all(np.iscomplex(state.data))
    text_content = state_title(state.shape,is_complex)
    state._len_max_row = len(str(state.shape[0]))
    for i in range(state.nnz):
        initial = space_symbol * (3 + state._len_max_row - len(str(state.row[i])))
        real_formatted = space4_symbol + "{: .16E}".format(np.real(state.data[i]))
        if is_complex:
            imag_formatted = space4_symbol + "{: .16E}".format(np.imag(state.data[i]))
            text_content += f'{initial}{state.row[i]}{real_formatted}{imag_formatted}\n'
        else:text_content += f'{initial}{state.row[i]}{real_formatted}\n'
    return text_content

def matrix2text(matrix):
    if not isinstance(matrix,coo_matrix):matrix = coo_matrix(matrix)  
    len_max_row = len(str(matrix.shape[0]))  
    is_complex = all(np.iscomplex(matrix.data))
    text_content = matrix_title(matrix.shape,is_complex)
    for i in range(len(matrix.row)):
        row_text = space_symbol * (3 + len_max_row - len(str(matrix.row[i]))) + str(matrix.row[i])
        col_text = space_symbol * (3 + len_max_row - len(str(matrix.col[i]))) + str(matrix.col[i])
        imag_formatted = space4_symbol +"{: .16E}".format(np.imag(matrix.data[i]))
        real_formatted = space4_symbol +"{: .16E}".format(np.real(matrix.data[i]))
        if is_complex:text_content += row_text + col_text + real_formatted + imag_formatted + '\n'
        else:text_content += row_text + col_text + real_formatted + '\n'
    return text_content

def control2text(tlist,amplitude):
    is_complex = all(np.iscomplex(amplitude))
    text_content = pulse_title(is_complex)
    for i in range(len(tlist)):
        t_formatted = "{: .16E}".format(tlist[i])
        if is_complex:
            real_formatted = "{: .16E}".format(np.real(amplitude[i]))
            imag_formatted = "{: .16E}".format(np.imag(amplitude[i]))
            text_content += t_formatted + space4_symbol + real_formatted+ imag_formatted + '\n'
        else:
            amp_formatted = "{: .16E}".format(amplitude[i])
            text_content += t_formatted + space4_symbol + amp_formatted + '\n'
    return text_content

def stateReader(file_name, max_dim, zero_base = True):
    data = np.loadtxt(file_name, usecols=(0, 1))
    if len(data.shape) == 1:
        ids = int(data[0])
        if not zero_base: ids -= 1
        vals = data[1]
        try:
            imags = np.loadtxt(file_name, usecols=(2))
            vals = np.complex128(vals)
            vals += 1j * imags
            state = np.zeros([max_dim, 1], dtype=np.complex128)
        except Exception:
            state = np.zeros([max_dim, 1])
        state[ids] += vals
        return state
    ids, vals = np.split(np.loadtxt(file_name, usecols=(0, 1)), 2, axis=1)
    ids = ids.T[0]
    if not zero_base:ids = [int(ids[i] - 1) for i in range(len(ids))]
    else:ids = [int(ids[i]) for i in range(len(ids))]
    vals = vals.T[0]
    try:
        imags = np.loadtxt(file_name, usecols=(2))
        vals = np.complex128(vals)
        vals += 1j * imags
        state = np.zeros([max_dim, 1], dtype=np.complex128)
    except Exception:
        state = np.zeros([max_dim, 1])
    for i in range(len(ids)):
        state[ids[i]][0] += vals[i]
    return state

def matrixReader(file_name, max_dim, zero_base = True):
    data = np.loadtxt(file_name, usecols=(0, 1))
    if len(data.shape) == 1:
        rows = int(data[0])
        cols = int(data[1])
        if not zero_base: 
            rows -= 1
            cols -= 1
        try:
            vals = np.loadtxt(file_name, usecols=(2, 3))
            vals = vals[0] + 1j * vals[1]
            mat = np.zeros([max_dim, max_dim], dtype=np.complex128)
        except Exception:
            vals = np.loadtxt(file_name, usecols=(2))
            mat = np.zeros([max_dim, max_dim])
        mat[rows][cols] += vals
        return mat
    rows, cols = np.split(np.loadtxt(file_name, usecols=(0, 1)), 2, axis=1)
    try:
        reals, imags = np.split(np.loadtxt(file_name, usecols=(2, 3)), 2, axis=1)
        vals = reals.T[0] + 1j * imags.T[0]
        mat = np.zeros([max_dim, max_dim], dtype=np.complex128)
    except Exception:
        vals = np.loadtxt(file_name, usecols=(2))
        vals = vals.T
        mat = np.zeros([max_dim, max_dim])
    rows = rows.T[0]
    cols = cols.T[0]
    if not zero_base:
        rows = [int(rows[i] - 1) for i in range(len(rows))]
        cols = [int(cols[i] - 1) for i in range(len(cols))]
    else:
        rows = [int(rows[i]) for i in range(len(rows))]
        cols = [int(cols[i]) for i in range(len(cols))]
    for i in range(len(rows)):
        mat[rows[i]][cols[i]] += vals[i]
    return mat

def controlReader(file_name):
    tlist, control = np.split(np.loadtxt(file_name, usecols=(0, 1)), 2, axis=1)
    tlist = tlist.T[0]
    control = control.T[0]
    return tlist,control
    try:
        control_imag = np.loadtxt(file_name, usecols=(2))
        control = np.complex128(control)
        control += 1j * control_imag
        return tlist,control
    except Exception:return tlist,control