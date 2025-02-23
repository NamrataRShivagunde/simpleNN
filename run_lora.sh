#!/bin/bash

# Define lists of hyperparameter values to iterate over
hidden_sizes=(64)
output_sizes=(2)
learning_rates=(0.15)
batch_sizes=(256)
num_epochs=(10)
rank=(16 12 8 4 2 1)

# Loop through each combination of hyperparameter values
for hidden_size in "${hidden_sizes[@]}"
do
    for output_size in "${output_sizes[@]}"
    do
        for learning_rate in "${learning_rates[@]}"
        do
            for batch_size in "${batch_sizes[@]}"
            do
                for num_epoch in "${num_epochs[@]}"
                do
                    for num_epoch in "${rank[@]}"
                    do
                        echo "Running script with LORA with hyperparameters: hidden_size=$hidden_size, output_size=$output_size, learning_rate=$learning_rate, batch_size=$batch_size, num_epochs=$num_epoch"
                        python main.py --hidden_size $hidden_size --output_size $output_size --learning_rate $learning_rate --batch_size $batch_size --num_epochs $num_epoch --add_lora --rank $rank
                    done
                done
            done
        done
    done
done