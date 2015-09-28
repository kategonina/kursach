# coding: utf-8

number = rand(1..100) 
num_guesses = 0

puts 'Я загадал случайное число от 1 до 100' 
puts 'Сможете отгадать его?'

loop do 
   print 'Ваш вариант: ' 
   guess = gets.chomp.to_i 
   num_guesses += 1
 
 unless guess == number 
   message = if guess > number 
                'Слишком большое' 
             else 
                'Слишком маленькое' 
             end 
   puts message 
 else 
   puts 'Вы угадали!' 
   puts "Вам потребовалось #{num_guesses} попыток" 
   exit 
 end 
end