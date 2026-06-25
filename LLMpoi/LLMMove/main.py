import argparse
import os
import sys

case_num = 999999

# Resolve paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')


def read_csv_lines(filePath):
    for enc in ['utf-8-sig', 'iso-8859-1', 'cp1252']:
        try:
            with open(filePath, 'r', encoding=enc) as f:
                lines = f.readlines()
            for line in lines[:5]:
                line.encode('utf-8')
            return lines
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    with open(filePath, 'r', encoding='utf-8', errors='replace') as f:
        return f.readlines()


def readTrain(filePath):
    longs = dict()
    pois = dict()
    lines = read_csv_lines(filePath)
    for line in lines[1:]:
        data = line.split(',')
        time, u, lati, longi, i, category = data[1], data[5], data[6], data[7], data[8], data[10]
        i = i.strip()
        u = u.strip()
        if i not in pois:
            pois[i] = {"latitude": lati.strip(), "longitude": longi.strip(), "category": category.strip()}
        if u not in longs:
            longs[u] = list()
        longs[u].append((i, time.strip()))
    return longs, pois


def readTest(filePath):
    recents = dict()
    pois = dict()
    targets = dict()
    traj2u = dict()
    lines = read_csv_lines(filePath)
    for line in lines[1:]:
        data = line.split(',')
        time, trajectory, u, lati, longi, i, category = data[1], data[3], data[5], data[6], data[7], data[8], data[10]
        i = i.strip()
        trajectory = trajectory.strip()
        if i not in pois:
            pois[i] = dict()
            pois[i]["latitude"] = lati.strip()
            pois[i]["longitude"] = longi.strip()
            pois[i]["category"] = category.strip()
        if trajectory not in traj2u:
            traj2u[trajectory] = u.strip()
        if trajectory not in recents:
            recents[trajectory] = list()
            recents[trajectory].append((i, time.strip()))
        else:
            if trajectory in targets:
                recents[trajectory].append(targets[trajectory])
            targets[trajectory] = (i, time.strip())
    return recents, pois, targets, traj2u


def getData(datasetName):
    dir_map = {'nyc': 'NYC', 'tky': 'TKY'}
    base = os.path.join(_DATA_DIR, dir_map[datasetName], '{}')
    trainPath = base.format('train.csv')  # data/NYC/NYC_train.csv → base = data/NYC/NYC_{}
    testPath = base.format('val.csv')
    # Fix: base is like /path/data/NYC/NYC_{}, so format gives /path/data/NYC/NYC_train.csv
    trainPath = os.path.join(_DATA_DIR, dir_map[datasetName], f'{dir_map[datasetName]}_train.csv')
    testPath = os.path.join(_DATA_DIR, dir_map[datasetName], f'{dir_map[datasetName]}_val.csv')

    if not os.path.exists(trainPath):
        raise FileNotFoundError(f"Training data not found: {trainPath}")
    if not os.path.exists(testPath):
        raise FileNotFoundError(f"Test data not found: {testPath}")

    print(f"  Train: {trainPath}")
    print(f"  Test:  {testPath}")

    longs, poiInfos = readTrain(trainPath)
    recents, testPoi, targets, traj2u = readTest(testPath)
    poiInfos.update(testPoi)

    valid_targets = {}
    skipped = 0
    for tid, gt in targets.items():
        if tid in recents and len(recents[tid]) >= 1:
            valid_targets[tid] = gt
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} trajectories with <2 check-ins")

    targets = dict(list(valid_targets.items())[:case_num])

    return longs, recents, targets, poiInfos, traj2u


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetName', type=str, choices=['nyc', 'tky'], default='nyc', help='nyc/tky')
    parser.add_argument('--cases', type=int, default=None,
                        help='Number of test trajectories to run. Default: all. Use --cases 1 for a quick test.')
    parser.add_argument('--llm', type=str, default='qwen3.7-plus',
                        help='LLM model name (e.g. qwen3.7-plus, deepseek-v4-flash, gpt-3.5-turbo)')
    parser.add_argument('--api-base', type=str, default='https://opencode.ai/zen/go/v1',
                        help='OpenAI-compatible API base URL')
    args = parser.parse_args()

    if args.cases is not None:
        case_num = args.cases

    data = getData(args.datasetName)

    out_dir = os.path.join(_SCRIPT_DIR, 'output')
    os.makedirs(os.path.join(out_dir, 'LLMMove', args.datasetName), exist_ok=True)
    res_dir = os.path.join(_SCRIPT_DIR, 'results')
    os.makedirs(res_dir, exist_ok=True)

    sys.path.insert(0, _SCRIPT_DIR)
    from models.LLMMove import LLMMove
    model = LLMMove(llm_model=args.llm, api_base=args.api_base, output_dir=out_dir)

    results = model.run(data, args.datasetName)
    results = 'ACC@1: {}, ACC@10: {}, MRR: {}, ValidRatio: {}'.format(results[0], results[1], results[2], results[3])
    resultPath = os.path.join(res_dir, f'LLMMove_{args.datasetName}')
    with open(resultPath, 'w') as file:
        file.write(results)
    print(f"Results saved to {resultPath}")
