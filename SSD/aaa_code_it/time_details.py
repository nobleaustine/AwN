import cProfile
import pstats
import re

def show_profile(func,kwargs):
    profiler = cProfile.Profile()
    profiler.runcall(func,**kwargs)
    stats = pstats.Stats(profiler)
    stats.strip_dirs()  # Remove the file path
    stats.sort_stats('cumulative')
    stats.print_stats(lambda x: re.search(r'(<built-in|lib/python|dist-packages|site-packages', x[1]) is None)
