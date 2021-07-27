from typing import List, Optional

import numpy as np

__all__ = [
    'impute_average',
]


def extract_sequences(ar: List[int]) -> List[List[int]]:
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
    mx = np.nanmean(x)
    my = np.nanmean(y)
    
    vals = (x - mx) * (y - my)
    n = x.size - np.isnan(vals).sum() - ddof
    
    return vals.sum() / n


def calculate_slope(x: np.ndarray, y: np.ndarray) -> float:
    return nancov(x, y, ddof=1) / np.nanvar(x, ddof=1)


def impute_average(
        x: np.ndarray,
        low: Optional[float] = None,
        high: Optional[float] = None,
        period: int = 10,
) -> np.ndarray:
    x_filled = []
    
    for row in x:
        row = row.copy()
        
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
                
        start = sequences[0][0]
        
        if start > 0:
            x0 = row[start]
            
            ox = np.arange(start, start + period)
            oy = row[ox]
            slope = calculate_slope(ox, oy)
            
            for i in range(start):
                row[i] = x0 + (i - start) * slope
                
        end = sequences[-1][-1]
        
        if end < x.shape[1] - 1:
            x1 = row[end]
            
            ox = np.arange(end - period, end) + 1
            oy = row[ox]
            slope = calculate_slope(ox, oy)
            
            for i in range(end + 1, x.shape[1]):
                row[i] = x1 + (i - end) * slope
                
        x_filled.append(row)
        
    x_filled = np.array(x_filled)
    
    if low is not None:
        x_filled[x_filled < low] = low
        
    if high is not None:
        x_filled[x_filled > high] = high

    return x_filled
