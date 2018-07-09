require 'torch'
require 'nn'
require 'optim'

-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
  dataset = 'video5',   -- indicates what dataset load to use (in data.lua)
  nThreads = 32,        -- how many threads to pre-fetch data
  batchSize = 32,      -- self-explanatory
  loadSize = 128,       -- when loading images, resize first to this size
  fineSize = 64,       -- crop this size from the loaded image 
  frameSize = 32,
  lr = 0.00002,          -- learning rate
  lr_decay = 1000,         -- how often to decay learning rate (in epoch's)
  lambda = 10,
  beta1 = 0.5,          -- momentum term for adam
  meanIter = 0,         -- how many iterations to retrieve for mean estimation
  saveIter = 3500,    -- write check point on this interval
  niter = 10,          -- number of iterations through dataset (epoch)
  ntrain = math.huge,   -- how big one epoch should be
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 0,            -- whether to use cudnn or not
  finetune = '',        -- if set, will load this network instead of starting from scratch
  preloadadversial = '../models/gtea_adversial_m2/iter0_net.t7', --load pretrained adversial for DFG-P gaze prior map generation
  name = 'gtea_gaze_prior',        -- the name of the experiment
  randomize = 1,        -- whether to shuffle the data file or not
  cropping = 'random',  -- options for data augmentation
  display_port = 8000,  -- port to push graphs
  display_id = 1,       -- window ID when pushing graphs
  mean = {0,0,0},
  data_root = '../dataset/',
  data_list = '../filelist/gtea_fulllist_train.txt',
  data_listmask = '../filelist/gtea_fulllist_train_mask.txt',  
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
if opt.gpu > 0 then
  require 'cunn'
  cutorch.setDevice(opt.gpu)
end

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- define the model
local net
local netgaze

if opt.finetune == '' then -- build network from scratch

  net = nn.Sequential()
  
  net:add(nn.View(-1, 1024, 1, 4, 4))
  net:add(nn.VolumetricFullConvolution(1024, 1024, 2,1,1))
  net:add(nn.VolumetricBatchNormalization(1024)):add(nn.ReLU(true))
  net:add(nn.VolumetricFullConvolution(1024, 512, 4,4,4, 2,2,2, 1,1,1))
  net:add(nn.VolumetricBatchNormalization(512)):add(nn.ReLU(true))
  net:add(nn.VolumetricFullConvolution(512, 256, 4,4,4, 2,2,2, 1,1,1))
  net:add(nn.VolumetricBatchNormalization(256)):add(nn.ReLU(true))
  net:add(nn.VolumetricFullConvolution(256, 128, 4,4,4, 2,2,2, 1,1,1))
  net:add(nn.VolumetricBatchNormalization(128)):add(nn.ReLU(true))
  net:add(nn.VolumetricFullConvolution(128,1, 4,4,4, 2,2,2, 1,1,1))

  net:add(nn.Squeeze())
  net:add(nn.View(opt.batchSize, opt.frameSize, -1))
  net:add(nn.Transpose({1,3}))  
  net:add(nn.SoftMax())
  net:add(nn.Log())
  net:add(nn.Transpose({1,3}))
  net:add(nn.View(opt.batchSize, 1, opt.frameSize, opt.fineSize,opt.fineSize)) 

  -- initialize the model
  local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
    end
  end
  net:apply(weights_init) -- loop over all layers, applying weights_init
  
  
  print('loading ' .. opt.preloadadversial)
  netgaze = torch.load(opt.preloadadversial)
  netgaze:remove(5)
  netgaze:remove(4)
  netgaze:remove(3)
  netgaze:remove(2)

  --save the first iteration of gaze prediction module
  paths.mkdir('../models/'.. opt.name)
  torch.save('../models/'.. opt.name .. '/iter0_net.t7', net:clearState())

  
else -- load in existing network
  --print('loading ' .. opt.finetune)
  --netgaze = torch.load(opt.finetune)
  --net = torch.load(opt.preloadadversial)
end

print('part1:')
print(netgaze)
print('part2:')
print(net)

-- define the loss
local criterion = nn.DistKLDivCriterion()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 1024,4,4)
local inputfake = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local target = torch.Tensor(opt.batchSize, 1, opt.frameSize, opt.fineSize, opt.fineSize)
local err

-- timers to roughly profile performance
local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship everything to GPU if needed
if opt.gpu > 0 then
  input = input:cuda()
  inputfake = inputfake:cuda()
  target = target:cuda()
 
  net:cuda()
  netgaze:cuda()
  criterion:cuda()
  
end

local parametersD, gradParametersD = net:getParameters()

-- optimization closure
-- the optimizer will call this function to get the gradients
local data_im
local maskTable

local fDx = function(x)
  gradParametersD:zero()

  -- fetch data
  data_tm:reset(); data_tm:resume()
  data_im, extraTable, maskTable = data:getBatch()
  data_tm:stop()

  -- ship to GPU
  inputfake:copy(data_im:select(3,1))
  target:copy(maskTable)

  input = netgaze:forward(inputfake)
  output = net:forward(input)
 
  -- forward/backwards real examples  
  err = criterion:forward(output, target)
  local df = criterion:backward(output, target)
  net:backward(input, df)

  
  return errD, gradParametersD
end



local counter = 0
local history = {}

-- parameters for the optimization
-- very important: you must only create this table once! 
-- the optimizer will add fields to this table (such as momentum)
local optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

-- train main loop
for epoch = 1,opt.niter do -- for each epoch
  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do -- for each mini-batch
    collectgarbage() -- necessary sometimes
    
    tm:reset()

    -- do one iteration
    optim.adam(fDx, parametersD, optimStateD)
    

    if counter % 10 == 0 then
      table.insert(history, {counter, err})
      --disp.plot(history, {win=opt.display_id+1, title=opt.name, labels = {"iteration", "err"}})
    end    
    
    counter = counter + 1
    
    print(('%s: Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
              .. '  Err: %.4f'):format(
            opt.name, epoch, ((i-1) / opt.batchSize),
            math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
            tm:time().real, data_tm:time().real,
            err and err or -1))

    -- save checkpoint
    -- :clearState() compacts the model so it takes less space on disk
    if counter % opt.saveIter == 0 then
      print('Saving ' .. opt.name .. '/iter' .. counter .. '_net.t7')
      paths.mkdir('../models/'.. opt.name)
      torch.save('../models/'.. opt.name .. '/iter' .. counter .. '_net.t7', netgaze:clearState())
      torch.save('../models/'.. opt.name .. '/iter' .. counter .. '_history.t7', history)
      
    end
  end
  
  -- decay the learning rate, if requested
  if opt.lr_decay > 0 and epoch % opt.lr_decay == 0 then
    opt.lr = opt.lr / 10
    print('Decreasing learning rate to ' .. opt.lr)

    -- create new optimState to reset momentum
    optimState = {
      learningRate = opt.lr,
      beta1 = opt.beta1,
    }
  end
end

print('trained successfully')
os.exit()

