search_dir="/home/schiavazza/data/hpe/vicon_dataset/processed/gaurvi"
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

    python3 ../save_dvs_frames.py --dvs_path ${input_static} --output_path ${output_static}
    python3 ../save_dvs_frames.py --dvs_path ${input_dynamic} --output_path ${output_dynamic} --time_window 0.002

done