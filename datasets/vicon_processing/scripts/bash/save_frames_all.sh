<<<<<<< Updated upstream
search_dir="/home/schiavazza/data/hpe/vicon_dataset/processed/gaurvi"
=======
search_dir="/home/iit.local/schiavazza/data/vicon_hpe/roberta"
>>>>>>> Stashed changes
echo "${search_dir}/*/"
for base_dir in ${search_dir}/*/   # list directories in the form "/tmp/dirname/"
do
    base_dir=${base_dir%*/}      # remove the trailing "/"
    echo "processing ${dir##*/}..."    # print everything after the final "/"

    atis_s="atis-s"
    atis_d="atis-d"
    atis_s_frames="atis_s_frames"
    atis_d_frames="atis_d_frames"

    input_static="${base_dir}/${atis_s}/"
    input_dynamic="${base_dir}/${atis_d}/"

    output_static="${base_dir}/${atis_s_frames}/"
    output_dynamic="${base_dir}/${atis_d_frames}/"

<<<<<<< Updated upstream
    type=${base_dir: -2 : 1}
=======
   type=${base_dir: -2 : 1}

    if ! [ -f ${output_static}/times.yml ]; then
        python3 ../save_dvs_frames.py --dvs_path ${input_static} --output_path ${output_static}
    fi
    if [ "$type" = "s" ]; then
        if ! [ -f ${output_dynamic}/times.yml ]; then
            python3 ../save_dvs_frames.py --dvs_path ${input_dynamic} --output_path ${output_dynamic} --time_window 0.02
        fi
    else
        if ! [ -f ${output_dynamic}/times.yml ]; then
            python3 ../save_dvs_frames.py --dvs_path ${input_dynamic} --output_path ${output_dynamic} --time_window 0.005
        fi
    fi
>>>>>>> Stashed changes

    python3 ../save_dvs_frames.py --dvs_path ${input_static} --output_path ${output_static}
    if [ "$type" = "s" ]; then
        python3 ../save_dvs_frames.py --dvs_path ${input_dynamic} --output_path ${output_dynamic} --time_window 0.01
    else
        python3 ../save_dvs_frames.py --dvs_path ${input_dynamic} --output_path ${output_dynamic} --time_window 0.002
    fi
done