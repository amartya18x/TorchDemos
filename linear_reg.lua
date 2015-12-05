-- Read CSV file
require 'optim'
-- Split string
function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
end


local filePath = 'housing.data'

-- Count number of rows and columns in file
local i = 0
for line in io.lines(filePath) do
  if i == 0 then
    COLS = #line:split(',')
  end
  i = i + 1
end

print(i)
local ROWS = i - 1  -- Minus 1 because of header

-- Read data from CSV to tensor
local csvFile = io.open(filePath, 'r')
local header = csvFile:read()

local data = torch.Tensor(ROWS, COLS)

local i = 0
for line in csvFile:lines('*l') do
  i = i + 1
  local l = line:split(',')
  for key, val in ipairs(l) do
    data[i][key] = val
  end
end

print(i)
csvFile:close()

-- Serialize tensor
local outputFilePath = 'train.th7'
torch.save(outputFilePath, data)

-- Deserialize tensor object
local restored_data = torch.load(outputFilePath)

-- Make test
print(data:size())
print(restored_data:size())


data = torch.load('train.th7')






sgd_params = {
   learningRate = 1e-6,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 1e-4
}

require 'nn'
model = nn.Sequential()                 -- define the container
ninputs = 13; noutputs = 1
model:add(nn.Linear(ninputs, noutputs)) -- define the only module



criterion = nn.MSECriterion()


x, dl_dx = model:getParameters()

feval = function(x_new)
   -- set x to x_new, if differnt
   -- (in this simple example, x_new will typically always point to x,
   -- so the copy is really useless)
   if x ~= x_new then
      x:copy(x_new)
   end

   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#data)[1] then _nidx_ = 1 end

   local sample = data[_nidx_]
   local target = sample[{ {14} }]      -- this funny looking syntax allows
   local inputs = sample[{ {1,13} }]    -- slicing of arrays.
   -- reset gradients (gradients are always accumulated, to accommodate 
   -- batch methods)
   dl_dx:zero()

   target = target:double()
   inputs = inputs:double()
   -- evaluate the loss function and its derivative wrt x, for that sample
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))
   --print (model)
   
   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end

for i = 1,1e4 do

   -- this variable is used to estimate the average loss
   current_loss = 0

   -- an epoch is a full loop over our training data
   for i = 1,(#data)[1] do

      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific
      
      _,fs = optim.sgd(feval,x,sgd_params)

      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.

      current_loss = current_loss + fs[1]
   end

   -- report average error on epoch
   current_loss = current_loss / (#data)[1]
   print('current loss = ' .. current_loss)

end
