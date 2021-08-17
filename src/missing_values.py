from typing import List, Optional, Tuple, Union

import numpy as np

__all__ = [
    'impute_average',
]


def extract_sequences(ar: List[int]) -> List[List[int]]:
    """Функция принимает список индексов (возрастающий, но с пропусками)
    и разбивает его на списки без пропусков.
    """
    
    sequences = []
    current_sequence = [ar[0]]
    
    for val in ar[1:]:
        if val - current_sequence[-1] == 1:
            current_sequence.append(val)
        
        else:
            sequences.append(current_sequence)
            current_sequence = [val]
            
    sequences.append(current_sequence)
            
    return sequences


def nancov(x: np.ndarray, y: np.ndarray, ddof: int = 1) -> float:
    """Ковариация с обработкой пропусков.
    """
    
    mx = np.nanmean(x)
    my = np.nanmean(y)
    
    vals = (x - mx) * (y - my)
    n = x.size - np.isnan(vals).sum() - ddof
    
    return np.nansum(vals) / n


def calculate_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Коэффициент наклона парной регрессии с обработкой пропусков. 
    """
    
    return nancov(x, y, ddof=1) / np.nanvar(x, ddof=1)


def impute_average(
        x: np.ndarray,
        lim: Union[None, str, Tuple[float, float]] = 'auto',
        impute_tails: bool = True,
        period: int = 7,
) -> np.ndarray:
    """Заполнение пропусков "по линейке" в каждой строке матрицы `x`.
    Отдельно обрабатывает пропуски "внутри" строки и по краям.
    
    1. Пропуски внутри строки (т.е. для которых имеются значения с
    обеих сторон) заполняются по линии, соединяющей эти имеющиеся
    значения по краям.
    2. Пропуски по краям строки заполняются (если `impute_tails=True`)
    от крайнего значения вдоль линии линейной регрессии, построенной по
    `period` крайним имеющимся значениям, если столько нашлось.
    
    Параметр `lim` отвечает за верхнюю и нижнюю границы значений переменных.
    Значения, выходящие за пределы, обрубаются по ним. Если `lim='auto'`,
    используется максимум и минимум по всем значениям матрицы `x`.
    """
    
    x = x.copy()
    
    if lim is None:
        low = None
        high = None
        
    elif lim == 'auto':
        low = np.min(x)
        high = np.max(x)
    
    else:
        low, high = lim
    
    flat = len(x.shape) == 1
    
    if flat:
        x = x.reshape(1, -1)
    
    x_filled = []
    
    for row in x:
        idx = np.where(~np.isnan(row))[0]
        
        if len(idx) <= 1:
            x_filled.append(row)
            continue
        
        sequences = extract_sequences(idx)
        
        for seq1, seq2 in zip(sequences, sequences[1:]):
            start = seq1[-1]
            end = seq2[0]
            assert end - start > 1
            
            x0 = row[start]
            x1 = row[end]
            slope = (x1 - x0) / (end - start)
            
            for i in range(start + 1, end):
                row[i] = x0 + (i - start) * slope
        
        if not impute_tails:
            continue
        
        start = sequences[0][0]
        
        if start > 0:
            x0 = row[start]
            
            ox = np.arange(start, start + period)
            oy = row[ox]
            slope = calculate_slope(ox, oy)
            
            for i in range(start):
                row[i] = x0 + (i - start) * slope
                
        end = sequences[-1][-1]
        
        if end < row.shape[0] - 1:
            x1 = row[end]
            
            ox = np.arange(end - period, end) + 1
            oy = row[ox]
            slope = calculate_slope(ox, oy)
            
            for i in range(end + 1, row.shape[0]):
                row[i] = x1 + (i - end) * slope
                
        x_filled.append(row)
        
    x_filled = np.array(x_filled)
    
    if low is not None:
        x_filled[x_filled < low] = low
        
    if high is not None:
        x_filled[x_filled > high] = high

    if flat:
        x_filled = x_filled.flatten()
        
    return x_filled
