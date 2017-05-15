--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

-- Heavily moidifed by Carl to make it simpler

-- Later modified by Mengmi to add in mask and image pyramid inputs 


require 'torch'
require 'image'
tds = require 'tds'
torch.setdefaulttensortype('torch.FloatTensor')
local class = require('pl.class')

local dataset = torch.class('dataLoader')

-- this function reads in the data files
function dataset:__init(args)
  for k,v in pairs(args) do self[k] = v end

  assert(self.frameSize > 0)

  if self.filenamePad == nil then
    self.filenamePad = 8
  end

  -- read text file consisting of frame directories and counts of frames
  self.data = tds.Vec()
  self.datamask = tds.Vec()
  self.datapym1 = tds.Vec()
  self.datapym2 = tds.Vec()
  self.datapym3 = tds.Vec()
  print('reading ' .. args.data_list)
  

  for line in io.lines(args.data_list) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    self.data:insert(split[1])
  end

  for line in io.lines(args.data_listmask) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    self.datamask:insert(split[1])
  end

  for line in io.lines(args.data_listpym1) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    self.datapym1:insert(split[1])
  end

  for line in io.lines(args.data_listpym2) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    self.datapym2:insert(split[1])
  end

  for line in io.lines(args.data_listpym3) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    self.datapym3:insert(split[1])
  end

  print('found ' .. #self.data .. ' videos')
  print('found ' .. #self.datamask .. ' videos')
  print('found ' .. #self.datapym1 .. ' videos')
  print('found ' .. #self.datapym2 .. ' videos')
  print('found ' .. #self.datapym3 .. ' videos')
end

function dataset:size()
  return #self.data
end

-- converts a table of samples (and corresponding labels) to a clean tensor
function dataset:tableToOutput(dataTable, extraTable, maskTable, pym1Table, pym2Table, pym3Table)
   local data, scalarLabels, labels, maskdata, pym1data, pym2data, pym3data

   --data
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 4)
   data = torch.Tensor(quantity, 3, self.frameSize, self.fineSize, self.fineSize)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
   end

   --mask
   local quantitymask = #maskTable
   assert(maskTable[1]:dim() == 4)
   maskdata = torch.Tensor(quantitymask, 1, self.frameSize, self.fineSize, self.fineSize)
   for i=1,#maskTable do
      maskdata[i]:copy(maskTable[i])
   end

   --pym1
   local quantitypym1 = #pym1Table
   assert(pym1Table[1]:dim() == 4)
   pym1data = torch.Tensor(quantitypym1, 3, self.frameSize, self.fineSize, self.fineSize)
   for i=1,#maskTable do
      pym1data[i]:copy(pym1Table[i])
   end

   --pym2
   local quantitypym2 = #pym2Table
   assert(pym2Table[1]:dim() == 4)
   pym2data = torch.Tensor(quantitypym2, 3, self.frameSize, self.fineSize, self.fineSize)
   for i=1,#pym2Table do
      pym2data[i]:copy(pym2Table[i])
   end

   --pym3
   local quantitypym3 = #pym3Table
   assert(pym3Table[1]:dim() == 4)
   pym3data = torch.Tensor(quantitypym3, 3, self.frameSize, self.fineSize, self.fineSize)
   for i=1,#pym3Table do
      pym3data[i]:copy(pym3Table[i])
   end
   

   return data, extraTable, maskdata, pym1data, pym2data, pym3data
end

-- sampler, samples with replacement from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local extraTable = {}
   local maskTable = {}
   local pym1Table = {}
   local pym2Table = {}
   local pym3Table = {}

   for i=1,quantity do
      local idx = torch.random(1, #self.data)

      local data_path = self.data_root .. '/' .. self.data[idx]
      local out = self:trainHook(data_path)
      table.insert(dataTable, out)
      table.insert(extraTable, self.data[idx])

      local maskdata_path = self.data_root .. '/' .. self.datamask[idx]
      local outmask = self:trainHookMengmiMask(maskdata_path)
      table.insert(maskTable, outmask)

      local pym1data_path = self.data_root .. '/' .. self.datapym1[idx]
      local outpym1 = self:trainHook(pym1data_path)
      table.insert(pym1Table, outpym1)

      local pym2data_path = self.data_root .. '/' .. self.datapym2[idx]
      local outpym2 = self:trainHook(pym2data_path)
      table.insert(pym2Table, outpym2)

      local pym3data_path = self.data_root .. '/' .. self.datapym3[idx]
      local outpym3 = self:trainHook(pym3data_path)
      table.insert(pym3Table, outpym3)

   end
   return self:tableToOutput(dataTable, extraTable, maskTable, pym1Table, pym2Table, pym3Table)
end

-- gets data in a certain range
function dataset:get(start_idx,stop_idx)
   local dataTable = {}
   local extraTable = {}
   local maskTable = {}
   local pym1Table = {}
   local pym2Table = {}
   local pym3Table = {}


   for idx=start_idx,stop_idx do
      local data_path = self.data_root .. '/' .. self.data[idx]
      local out = self:trainHook(data_path)
      table.insert(dataTable, out)
      table.insert(extraTable, self.data[idx])

      local maskdata_path = self.data_root .. '/' .. self.datamask[idx]
      local outmask = self:trainHookMengmiMask(maskdata_path)
      table.insert(maskTable, outmask)

      local pym1data_path = self.data_root .. '/' .. self.datapym1[idx]
      local outpym1 = self:trainHook(pym1data_path)
      table.insert(pym1Table, outpym1)

      local pym2data_path = self.data_root .. '/' .. self.datapym2[idx]
      local outpym2 = self:trainHook(pym2data_path)
      table.insert(pym2Table, outpym2)

      local pym3data_path = self.data_root .. '/' .. self.datapym3[idx]
      local outpym3 = self:trainHook(pym3data_path)
      table.insert(pym3Table, outpym3)

   end
   return self:tableToOutput(dataTable, extraTable, maskTable, pym1Table, pym2Table, pym3Table)

end

-- function to load the image, jitter it appropriately (random crops etc.)
function dataset:trainHook(path)
  collectgarbage()

  local oW = self.fineSize
  local oH = self.fineSize 
  local h1
  local w1

  local out = torch.zeros(3, self.frameSize, oW, oH)

  local ok,input = pcall(image.load, path, 3, 'float') 
  if not ok then
     print('warning: failed loading: ' .. path)
     return out
  end

  local count = input:size(2) / opt.loadSize
  local t1 = 1
  
  for fr=1,self.frameSize do
    local off 
    if fr <= count then 
      off = (fr+t1-2) * opt.loadSize+1
    else
      off = (count+t1-2)*opt.loadSize+1 -- repeat the last frame
    end
    local crop = input[{ {}, {off, off+opt.loadSize-1}, {} }]
    out[{ {}, fr, {}, {} }]:copy(image.scale(crop, opt.fineSize, opt.fineSize))
  end

  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

  -- subtract mean
  for c=1,3 do
    out[{ c, {}, {} }]:add(-self.mean[c])
  end

  return out
end

-- function to load the image, jitter it appropriately (random crops etc.) but for 1 dim mask
function dataset:trainHookMengmiMask(path)
  collectgarbage()

  local oW = self.fineSize
  local oH = self.fineSize 
  local h1
  local w1

  local out = torch.zeros(1, self.frameSize, oW, oH)

  local ok,input = pcall(image.load, path, 1, 'float') 
  if not ok then
     print('warning: failed loading: ' .. path)
     return out
  end

  local count = input:size(2) / opt.loadSize
  local t1 = 1
  
  for fr=1,self.frameSize do
    local off 
    if fr <= count then 
      off = (fr+t1-2) * opt.loadSize+1
    else
      off = (count+t1-2)*opt.loadSize+1 -- repeat the last frame
    end
    local crop = input[{ {}, {off, off+opt.loadSize-1}, {} }]
    out[{ {}, fr, {}, {} }]:copy(image.scale(crop, opt.fineSize, opt.fineSize))
  end

  --out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

  -- subtract mean
  --for c=1,3 do
    --out[{ c, {}, {} }]:add(-self.mean[c])
  --end

  return out
end

-- data.lua expects a variable called trainLoader
trainLoader = dataLoader(opt)
