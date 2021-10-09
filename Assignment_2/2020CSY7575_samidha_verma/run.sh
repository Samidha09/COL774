#./run.sh 1 <path_of_train_data> <path_of_test_data>  <part_num>
#./run.sh 2 <path_of_train_data> <path_of_test_data> <binary_or_multi_class> <part_num>
if [ $1 -eq 1 ]
then
    python ./Q1/Naive_Bayes.py $2 $3 $4
elif [ $1 -eq 2 ]
then
   if [ $4 -eq 0 ]
   then
        python ./Q2/Binary_Classification.py $2 $3 $5
   elif [ $4 -eq 1 ]
   then
        python ./Q2/Multi_Class_Classification.py $2 $3 $5
   else
    echo "The fourth argument can be 0 (for binary classification) or 1 (for multiclass classification)"
   fi
else
   echo "The first argument should be either 1 or 2"
fi