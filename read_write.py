import title_routine,numpy as np
from scipy.sparse import coo_matrix,coo_array

def state2text(state):
    if not isinstance(state,coo_array):state=coo_array(state)
    is_complex = all(np.iscomplex(state.data))
    text_content = title_routine.state_title(state.shape,is_complex)
    state._len_max_row = len(str(state.shape[0]))
    for i in range(state.nnz):
        initial = title_routine.space_symbol * (3 + state._len_max_row - len(str(state.row[i])))
        real_formatted = title_routine.space4_symbol + "{: .16E}".format(np.real(state.data[i]))
        if is_complex:
            imag_formatted = title_routine.space4_symbol + "{: .16E}".format(np.imag(state.data[i]))
            text_content += f'{initial}{state.row[i]}{real_formatted}{imag_formatted}\n'
        else:text_content += f'{initial}{state.row[i]}{real_formatted}\n'
    return text_content

def matrix2text(matrix):
    if not isinstance(matrix,coo_matrix):matrix = coo_matrix(matrix)  
    len_max_row = len(str(matrix.shape[0]))  
    is_complex = all(np.iscomplex(matrix.data))
    text_content = title_routine.matrix_title(matrix.shape,is_complex)
    for i in range(len(matrix.row)):
        row_text = title_routine.space_symbol * (3 + len_max_row - len(str(matrix.row[i]))) + str(matrix.row[i])
        col_text = title_routine.space_symbol * (3 + len_max_row - len(str(matrix.col[i]))) + str(matrix.col[i])
        imag_formatted = title_routine.space4_symbol +"{: .16E}".format(np.imag(matrix.data[i]))
        real_formatted = title_routine.space4_symbol +"{: .16E}".format(np.real(matrix.data[i]))
        if is_complex:text_content += row_text + col_text + real_formatted + imag_formatted + '\n'
        else:text_content += row_text + col_text + real_formatted + '\n'
    return text_content

def control2text(tlist,amplitude):
    is_complex = all(np.iscomplex(amplitude))
    text_content = title_routine.pulse_title(is_complex)
    for i in range(len(tlist)):
        t_formatted = "{: .16E}".format(tlist[i])
        if is_complex:
            real_formatted = "{: .16E}".format(np.real(amplitude[i]))
            imag_formatted = "{: .16E}".format(np.imag(amplitude[i]))
            text_content += t_formatted + title_routine.space4_symbol + real_formatted+ imag_formatted + '\n'
        else:
            amp_formatted = "{: .16E}".format(amplitude[i])
            text_content += t_formatted + title_routine.space4_symbol + amp_formatted + '\n'
    return text_content

def stateReader(file_name, max_dim):
    ids, vals = np.split(np.loadtxt(file_name, usecols=(0, 1)), 2, axis=1)
    ids = ids.T[0]
    ids = [int(ids[i] - 1) for i in range(len(ids))]
    vals = vals.T[0]
    try:
        imags = np.split(np.loadtxt(file_name, usecols=(2)), 1, axis=1)
        vals += 1j * imags.T[0]
        state = np.zeros([max_dim, 1], dtype=np.complex128)    
    except Exception:
        state = np.zeros([max_dim, 1])
    for i in range(len(ids)):
        state[ids[i]][0] += vals[i]
    return state

def matrixReader(file_name, max_dim):
    rows, cols = np.split(np.loadtxt(file_name, usecols=(0, 1)), 2, axis=1)
    try:
        reals, imags = np.split(np.loadtxt(file_name, usecols=(2, 3)), 2, axis=1)
        vals = reals.T[0] + 1j * imags.T[0]
        state = np.zeros([max_dim, max_dim], dtype=np.complex128)
    except Exception:
        vals = np.loadtxt(file_name, usecols=(2))
        vals = vals.T
        state = np.zeros([max_dim, max_dim])
    rows = rows.T[0]
    cols = cols.T[0]
    rows = [int(rows[i] - 1) for i in range(len(rows))]
    cols = [int(cols[i] - 1) for i in range(len(cols))]
    for i in range(len(rows)):
        state[rows[i]][cols[i]] += vals[i]
    return state

def controlReader(file_name):
    tlist, control = np.split(np.loadtxt(file_name, usecols=(0, 1)), 2, axis=1)
    tlist = tlist.T[0]
    control = control.T[0]
    try:
        control_imag = np.split(np.loadtxt(file_name, usecols=(2)), 1, axis=1)
        control += 1j * control_imag.T[0]
        return tlist,control
    except Exception:return tlist,control