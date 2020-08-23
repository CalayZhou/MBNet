function demo_test()

dtDir='../data/result';

%% specify path of groundtruth annotaions
gtDir = './improve_annotations_liu';

%% evaluate detection results
kaist_eval_full(dtDir, gtDir, true, true);

end
