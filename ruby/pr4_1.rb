
nums = [3,5,6,8,10,4,4,46,4,5]  
first, last = nums.slice(0), nums.slice(-1)
s_elem = nums.find_all {|n|  n > first && n < last}.inject("[ ]") {|r, n|  r = n}
s_elem_index = (nums.rindex s_elem)-1
puts "#{s_elem_index} => #{s_elem}"
