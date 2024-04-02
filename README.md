# Graph-of-Gradient
Example result on only a single demogrpahic: race.

For the same parameters (tuned after grid search), specifically, learning rate 3e-3, the results are:  
(For overall acc/ndcg: [@10, @20]. For ethnicity_acc/ndcg: Row ['WHITE', 'BLACK', 'HISPANIC'], Column [@10, @20].)

Normal:

Final test_acc:[0.17643908 0.26110085], test_ndcg:[0.26656379 0.27345939]

Normal test_ethnicity_acc:[[0.17065003 0.25612097]
 [0.20209396 0.28453418]
 [0.17187903 0.25401088]]

Fair:

Final test_acc:[0.23032734 0.32407207], test_ndcg:[0.34178205 0.34083014],

Fair test_ethnicity_acc:[[0.22541894 0.31805558]
 [0.25179777 0.34817921]
 [0.22711514 0.32526572]]
