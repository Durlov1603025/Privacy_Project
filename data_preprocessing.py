!pip install git+https://github.com/MetaGuard/xror.git#egg=xror

from xror import XROR

# Load the file using XROR
file = XROR("2322d53c-42b6-4083-bc69-e3e026282a89.xror")

with open('2322d53c-42b6-4083-bc69-e3e026282a89.xror', 'rb') as f:
    fiile = f.read()
xror = XROR.unpack(fiile)


import pandas as pd

# Assuming the frames are stored in 'xror['frames']'
frames_data = xror.data['frames']  # Extract frames

# Convert to DataFrame
df = pd.DataFrame(frames_data)

# Save to CSV
df.to_csv('/content/original_data.csv', index=False)

print("Frames data saved to CSV!")