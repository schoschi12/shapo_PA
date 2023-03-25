import simnet
import pickle
import zstandard as zstd
import simnet.lib.datapoint as datapoint
from simnet.lib.datapoint import LocalReadHandle
import operator

'''
with open("data/gts/real_test/results_real_test_scene_1_0000.pkl", 'rb') as pickle_file:
    content = pickle.load(pickle_file)
print(content)

with open("data/Transfer/train/2A6u7Bi4kMwskBRLkfz7Qd.pickle.zstd", 'rb') as pickle_file:
    content = pickle.load(pickle_file)
print(content)
'''
dataset_path = "data/Transfer/train"
handles = []
dataset = LocalReadHandle(dataset_path)
# print('datset_path', os.path.abspath(self.dataset_path), '\ncontent', os.listdir(self.dataset_path))
for path in dataset.dataset_path.glob('*.pickle.zstd'):
    # for path in self.dataset_path.glob('*.pkl'):
    uid = path.name.partition('.')[0]
    handles.append(LocalReadHandle(dataset.dataset_path, uid))
sorted = sorted(handles, key=operator.attrgetter('uid'))
print(sorted)


with open('data/Transfer/train/2A6u7Bi4kMwskBRLkfz7Qd.pickle.zstd', 'rb') as f:
    compressed_data = f.read()

dctx = zstd.ZstdDecompressor()
data_bytes = dctx.decompress(compressed_data)
data = pickle.loads(data_bytes)

print(data)