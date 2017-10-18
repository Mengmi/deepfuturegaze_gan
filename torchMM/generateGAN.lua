require 'torch'
require 'nn'
require 'image'
require 'cunn'
--require 'cudnn'

opt = {
  
  model = '../models/gtea_adversial_m2/iter0_net.t7',
  dataset = 'video2',   -- indicates what dataset load to use (in data.lua)
  nThreads = 32,        -- how many threads to pre-fetch data
  batchSize = 32,      -- self-explanatory
  loadSize = 128,       -- when loading images, resize first to this size
  fineSize = 64,       -- crop this size from the loaded image 
  frameSize = 32,
  lr = 0.0002,          -- learning rate
  lr_decay = 1000,         -- how often to decay learning rate (in epoch's)
  lambda = 10,
  beta1 = 0.5,          -- momentum term for adam
  meanIter = 0,         -- how many iterations to retrieve for mean estimation
  saveIter = 1000,    -- write check point on this interval
  niter = 100,          -- number of iterations through dataset (epoch)
  ntrain = math.huge,   -- how big one epoch should be
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 0,            -- whether to use cudnn or not
  finetune = '',        -- if set, will load this network instead of starting from scratch
  name = 'gtea_adversial_m2',        -- the name of the experiment
  randomize = 1,        -- whether to shuffle the data file or not
  cropping = 'random',  -- options for data augmentation
  display_port = 8000,  -- port to push graphs
  display_id = 1,       -- window ID when pushing graphs
  mean = {0,0,0},
  data_root = '../dataset/',
  data_list = '../filelist/gtea_fulllist_test.txt',
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
cutorch.setDevice(opt.gpu)

net  = torch.load(opt.model) 
print("Net loaded successfully!")

net:evaluate()
net:cuda()
if opt.cudnn > 0 then
  require 'cudnn'
  net = cudnn.convert(net, cudnn)
end 


print('Generator:')
print(net)


-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

if opt.gpu > 0 then
  input = input:cuda()
end

-- generate inputs
local data_im
data_im = data:getBatch()
input:copy(data_im:select(3,1))

-- forward once
local gen = net:forward(input)
local video = net.modules[3].output[1]:float()
local mask = net.modules[3].output[2]:float()
local static = net.modules[3].output[3]:float()
local mask = mask:repeatTensor(1,3,1,1,1)

function WriteGif(filename, movie)
  for fr=1,movie:size(3) do
    image.save(filename ..  '.' .. string.format('%08d', fr) .. '.png', image.toDisplayTensor(movie:select(3,fr)))
  end
  cmd = "ffmpeg -f image2 -i " .. filename .. ".%08d.png -y " .. filename
  print('==> ' .. cmd)
  sys.execute(cmd)
  for fr=1,movie:size(3) do
    os.remove(filename .. '.' .. string.format('%08d', fr) .. '.png')
  end
end

paths.mkdir('../vis/')
--[[paths.mkdir('../vis/gen2/')
paths.mkdir('../vis/video2/')
paths.mkdir('../vis/videomask2/')
paths.mkdir('../vis/mask2/')
paths.mkdir('../vis/static2/')
paths.mkdir('../vis/datagt2/')

WriteGif('../vis/gen2/gen2.gif', gen) 
WriteGif('../vis/video2/video2.gif', video) 
WriteGif('../vis/videomask2/videomask2.gif', torch.cmul(video, mask))
WriteGif('../vis/mask2/mask2.gif', mask)
WriteGif('../vis/static2/static2.gif', static)
WriteGif('../vis/datagt2/datagt2.gif', data_im)]]--


WriteGif('../vis/gen2.gif', gen) 
WriteGif('../vis/video2.gif', video) 
WriteGif('../vis/videomask2.gif', torch.cmul(video, mask))
WriteGif('../vis/mask2.gif', mask)
WriteGif('../vis/static2.gif', static)
WriteGif('../vis/datagt2.gif', data_im)

print('done')

