import torch
import time


def timer_cpu(func):
    """Декоратор-таймер для cpu"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time 
    return wrapper


def timer_gpu(func):
    """Декоратор-таймер для gpu"""
    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(True)
        end_event = torch.cuda.Event(True)
        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        return result, start_event.elapsed_time(end_event)  
    return wrapper


def prepare_data(device):
    A = torch.randn(64, 1024, 1024, device=device)
    A_2 = torch.randn(64, 1024, 1024, device=device)
    B = torch.randn(128, 512, 512, device=device)
    B_2 = torch.randn(128, 512, 512, device=device)
    C = torch.randn(256, 256, 256, device=device)
    C_2 = torch.randn(256, 256, 256, device=device)
    return (A, A_2), (B, B_2), (C, C_2)


@timer_cpu
def matrix_prod_cpu(A, B):
    return torch.matmul(A, B)


@timer_gpu
def matrix_prod_gpu(A, B):
    return torch.matmul(A, B)


@timer_cpu
def matrix_sum_cpu(A, B):
    return A + B


@timer_gpu
def matrix_sum_gpu(A, B):
    return A + B


@timer_cpu
def matrix_prod_elem_cpu(A, B):
    return A * B


@timer_gpu
def matrix_prod_elem_gpu(A, B):
    return A * B


@timer_cpu
def matrix_transpose_cpu(A, B):
    return A.T, B.T


@timer_gpu
def matrix_transpose_gpu(A, B):
    return A.T, B.T


@timer_cpu
def matrix_sum_elem_cpu(A, B):
    return sum(A), sum(B)


@timer_gpu
def matrix_sum_elem_gpu(A, B):
    return sum(A), sum(B)


def log_result(cpu_times, gpu_times):
    time_matrix_sum_cpu = (sum(cpu_times[0:3])/3)
    time_matrix_sum_gpu = sum(gpu_times[0:3])/3

    time_matrix_prod_cpu = sum(cpu_times[3:6])/3
    time_matrix_prod_gpu = sum(gpu_times[3:6])/3

    time_matrix_sum_elem_cpu = sum(cpu_times[6:9])/3
    time_matrix_sum_elem_gpu = sum(gpu_times[6:9])/3

    time_matrix_prod_elem_cpu = sum(cpu_times[9:12])/3
    time_matrix_prod_elem_gpu = sum(gpu_times[9:12])/3

    time_matrix_transpose_cpu = sum(cpu_times[12::])/3
    time_matrix_transpose_gpu = sum(gpu_times[12::])/3
    print(
        f"\nОперация   |  CPU(мс)  |  GPU(мс)  |  Ускорение\n" +
        f"\nМатричное сложение   |  {time_matrix_sum_cpu:.3f}  | {time_matrix_sum_gpu:.3f}  |  {(time_matrix_sum_cpu/time_matrix_sum_gpu):.3f}x\n" +
        f"\nМатричное умножение   |  {time_matrix_prod_cpu:.3f}  | {time_matrix_prod_gpu:.3f}  |  {(time_matrix_prod_cpu/time_matrix_prod_gpu):.3f}x\n" +
        f"\nПоэлементное сложение   |  {time_matrix_sum_elem_cpu:.3f}  | {time_matrix_sum_elem_gpu:.3f}  |  {(time_matrix_sum_elem_cpu/time_matrix_sum_elem_gpu):.3f}x\n" +
        f"\nПоэлементное умножение   |  {time_matrix_prod_elem_cpu:.3f}  | {time_matrix_prod_elem_gpu:.3f}  |  {(time_matrix_prod_elem_cpu/time_matrix_prod_elem_gpu):.3f}x\n" +
        f"\nТранспонирование   |  {time_matrix_transpose_cpu:.3f}  | {time_matrix_transpose_gpu:.3f}  |  {(time_matrix_transpose_cpu/time_matrix_transpose_gpu):.3f}x\n"
        )


if __name__ == "__main__":
    (A_cpu, A_2_cpu), (B_cpu, B_2_cpu), (C_cpu, C_2_cpu) = prepare_data("cpu")
    (A_gpu, A_2_gpu), (B_gpu, B_2_gpu), (C_gpu, C_2_gpu) = prepare_data("cuda")
    cpu_times = []
    gpu_times = []
    cpu_funcs = [matrix_sum_cpu, matrix_prod_cpu, matrix_sum_elem_cpu, matrix_prod_elem_cpu, matrix_transpose_cpu]
    gpu_funcs = [matrix_sum_gpu, matrix_prod_gpu, matrix_sum_elem_gpu, matrix_prod_elem_gpu, matrix_transpose_gpu]
    for function in cpu_funcs:
        result, time_elapsed1 = function(A_cpu, A_2_cpu)
        result, time_elapsed2 = function(B_cpu, B_2_cpu)
        result, time_elapsed3 = function(C_cpu, C_2_cpu)
        cpu_times.extend([time_elapsed1*1000, time_elapsed2*1000, time_elapsed3*1000])
    for function in gpu_funcs:
        result, time_elapsed1 = function(A_gpu, A_2_gpu)
        result, time_elapsed2 = function(B_gpu, B_2_gpu)
        result, time_elapsed3 = function(C_gpu, C_2_gpu)
        gpu_times.extend([time_elapsed1, time_elapsed2, time_elapsed3])
    log_result(cpu_times, gpu_times)
