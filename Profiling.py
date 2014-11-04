import line_profiler as P
import test_accuracy_tagger as t
# mport HMMgenerator as gen
# import IterativeTraining

profiler = P.LineProfiler()
profiler.add_function(t.run)

ftrain = '../../Data/Lassy/lassy.train.spec1'
feval = '../../Data/Lassy/lassy.dev.spec1'

profiler.run('t.run(ftrain, feval)')
profiler.print_stats()
