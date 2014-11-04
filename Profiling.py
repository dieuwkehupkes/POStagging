import line_profiler as P
import Test
import test_accuracy_tagger as t
import HMMgenerator as gen
import IterativeTraining

profiler = P.LineProfiler()
profiler.add_module(IterativeTraining)
profiler.add_module(gen)
profiler.add_function(gen.HMM2_generator.get_lexicon_counts)
profiler.add_function(gen.HMM2_generator.get_lexicon_from_file)
profiler.add_function(t.run)
profiler.add_function(gen.HMM2_generator.get_lexicon_counts)
profiler.add_function(gen.HMM2_generator.get_trigrams_from_file)

ftrain = '../../Data/Lassy/lassy.train.spec1'
feval = '../../Data/Lassy/lassy.dev.spec1'

T = Test.Test()
profiler.run('t.run(ftrain, feval)')
profiler.print_stats()
