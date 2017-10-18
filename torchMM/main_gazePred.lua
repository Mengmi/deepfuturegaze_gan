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
  saveIter = 1000,    -- write check point on this interval
  niter = 100,          -- number of iterations through dataset (epoch)
  ntrain = math.huge,   -- how big one epoch should be
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 0,            -- whether to use cudnn or not
  finetune = '',        -- if set, will load this network instead of starting from scratch
  preloadadversial = '../models/gtea_adversial_m2/iter0_net.t7', --load pretrained adversial for frame prediction
  name = 'gtea_gaze_m16',        -- the name of the experiment
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

  netgaze = nn.Sequential()
  netgaze:add(nn.VolumetricConvolution(3,128, 3,3,3, 1,1, 1, 1,1,1))
  netgaze:add(nn.ReLU(true))
  netgaze:add(nn.VolumetricConvolution(128,256, 4,4,4, 2,2,2, 1,1,1))
  netgaze:add(nn.ReLU(true))

  netgaze:add(nn.VolumetricConvolution(256, 256, 3,3,3, 1,1,1, 1,1,1))  
  netgaze:add(nn.ReLU(true))
  netgaze:add(nn.VolumetricConvolution(256, 256, 3,3,3, 1,1,1, 1,1,1))  
  netgaze:add(nn.ReLU(true))
  
  mask_out = nn.VolumetricFullConvolution(256,1, 4,4,4, 2,2,2, 1,1,1)
  netgaze:add(mask_out)
  netgaze:add(nn.ReLU(true))

  --normalization of each saliency map and get ready for kld loss 
  netgaze:add(nn.Squeeze())
  netgaze:add(nn.View(opt.batchSize, opt.frameSize, -1))
  netgaze:add(nn.Transpose({1,3}))  
  netgaze:add(nn.SoftMax())
  netgaze:add(nn.Log())
  netgaze:add(nn.Transpose({1,3}))
  netgaze:add(nn.View(opt.batchSize, 1, opt.frameSize, opt.fineSize,opt.fineSize)) 

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
  netgaze:apply(weights_init) -- loop over all layers, applying weights_init
  
  mask_out.weight:normal(0, 0.01)
  mask_out.bias:fill(0)

  --load future frame generation module
  print('loading ' .. opt.preloadadversial)
  net = torch.load(opt.preloadadversial)
  net:evaluate()

  --save the first iteration of gaze prediction module
  paths.mkdir('../models/'.. opt.name)
  torch.save('../models/'.. opt.name .. '/iter0_net.t7', netgaze:clearState())

else -- load in existing network
  print('loading ' .. opt.finetune)
  netgaze = torch.load(opt.finetune)
  net = torch.load(opt.preloadadversial)
end

print('MengmiModel:')
print(netgaze)
print('Generator:')
print(net)

-- define the loss
local criterion = nn.DistKLDivCriterion()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.frameSize, opt.fineSize, opt.fineSize)
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
  
  net:evaluate()
  net:cuda()
  netgaze:cuda()
  criterion:cuda()
  
end

local parametersD, gradParametersD = netgaze:getParameters()

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

  --print(data_im:size())
  --print(maskTable:size())
  
  -- ship to GPU
  inputfake:copy(data_im:select(3,1))
  target:copy(maskTable)
  
  input = net:forward(inputfake)
   
  -- forward/backwards real examples
  output = netgaze:forward(input)
  --print(output:size())
  err = criterion:forward(output, target)
  local df = criterion:backward(output, target)
  netgaze:backward(input, df)

  
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

