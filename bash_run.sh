if [ $# -lt 1 ]; then
    echo "Please provide test data path"
    echo "** Usage: $0 <path_to_test_data>"
    exit 1
fi

python3 main.py --test_data $1 --debug

python3 main.py --test_data $1  --debug\
        --normalize_data

python3 main.py --test_data $1  --debug\
        --norm_type z_norm

python3 main.py --test_data $1  --debug\
    --algorithm backward_elimination

python3 main.py --test_data $1 --debug\
        --normalize_data --algorithm backward_elimination

python3 main.py --test_data $1  --debug\
        --norm_type z_norm --algorithm backward_elimination


