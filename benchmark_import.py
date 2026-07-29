import timeit
from audio_backend import get_sounddevice, get_soundfile

def run_benchmark():
    n_runs = 1000000

    start = timeit.default_timer()
    for _ in range(n_runs):
        get_sounddevice()
    end = timeit.default_timer()
    sd_time = end - start

    start = timeit.default_timer()
    for _ in range(n_runs):
        get_soundfile()
    end = timeit.default_timer()
    sf_time = end - start

    print(f"Time for get_sounddevice (1M calls): {sd_time:.4f}s")
    print(f"Time for get_soundfile (1M calls): {sf_time:.4f}s")

if __name__ == "__main__":
    run_benchmark()
