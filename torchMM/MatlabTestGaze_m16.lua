require 'torch'
require 'nn'
require 'image'
require 'cunn'
local matio = require 'matio'
--require 'cudnn'

opt = {
  model = '../models/gtea_gaze_m16/iter0_net.t7',
  dataset = 'video5',   -- indicates what dataset load to use (in data.lua)
  nThreads = 0,        -- how many threads to pre-fetch data
  batchSize = 32,      -- self-explanatory
  batchNum = 1,        --number of batches for testing: total number of images/batchsize
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
  name = 'gtea_gaze_m16',        -- the name of the experiment
  randomize = 0,        -- whether to shuffle the data file or not
  cropping = 'random',  -- options for data augmentation
  display_port = 8000,  -- port to push graphs
  display_id = 1,       -- window ID when pushing graphs
  mean = {0,0,0},
  data_root = '../dataset/',
  data_list = '../filelist/gtea_fulllist_test.txt',
  data_listmask = '../filelist/gtea_fulllist_test_mask.txt',
  resultdir = '../results/gtea_gaze_m16/', --where to store predicted saliency maps
  saveprefix = 'gtea_gaze_m16',
  --futuredir = '../results/gtea_gaze_m1_future/',
  --_current
  testcurrent = 0,
}

paths.mkdir(opt.resultdir)
--paths.mkdir(opt.futuredir)

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
cutorch.setDevice(opt.gpu)

--local net
local netgaze

if opt.testcurrent == 0 then
   net  = torch.load('../models/gtea_adversial_m2/iter0_net.t7')
end

netgaze = torch.load(opt.model)
print("Net loaded successfully!")

netgaze:evaluate()
netgaze:cuda()

if opt.testcurrent == 0 then
	net:evaluate()
	net:cuda()
end


if opt.cudnn > 0 then
  require 'cudnn'
  net = cudnn.convert(net, cudnn)
end 


print(netgaze)


-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.frameSize, opt.fineSize, opt.fineSize)
local inputfake = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
--local target = torch.Tensor(opt.batchSize, 1, opt.frameSize, opt.fineSize, opt.fineSize)


if opt.gpu > 0 then
  inputfake = inputfake:cuda()
  input = input:cuda()
  --target:cuda()
end

-- generate inputs
local data_im
local maskTable

----------------------------------------- test and put into matlab for evaluation ---------------------------------------------------
for i = 1, math.min(opt.batchNum, opt.ntrain), opt.batchSize do -- for each mini-batch

   print('processing....................................................')
   print('i = ' .. i)

   
   data_im, extraTable = data:getBatch()

   if opt.testcurrent == 0 then
        inputfake:copy(data_im:select(3,1))
        input = net:forward(inputfake)
   end
   
   if opt.testcurrent == 1 then
        input:copy(data_im)
   end
   
   mask = netgaze:forward(input)
   mask = mask:double()
      
   --write to matlab.mat file for evaluation using AUC, and AAE; save each saliency map per batch
   for k = 1,mask:size(1) do
      for j = 1,mask:size(3) do
         matio.save(opt.resultdir .. opt.saveprefix .. '_' .. (i+k-1) .. '_' .. j .. '.mat',mask[{{k},{},{j},{},{}}])
         --matio.save(opt.futuredir .. opt.saveprefix .. '_' .. (i+k-1) .. '_' .. j .. '.mat',oo[{{k},{},{j},{},{}}]  )
      end
   end

end
print('test generation done')


