#!/usr/bin/ruby -W0
# Find the hive with the lowest load
# Uses SSH and `uptime`.
# W0 because the hives have a world-writable, but read-only directory 
# in $PATH

HIVE_COUNT = 28
BANNED = [23, 25] # hosts that are broke

class HostLoad
  include Comparable

  attr_accessor :uptimes, :hostname

  def initialize(uptimes, hostname)
    @uptimes = uptimes
    @hostname = hostname
  end

  def <=> other
    self.to_f <=> other.to_f
  end

  def to_f
    @uptimes[0]
  end

  def to_s
    h = sprintf "%-23s", @hostname
    u = @uptimes.map{|u| sprintf "%06.3f", u}.join ", "
    "#{h}\t\t#{u}"
  end
end

def get_uptime(hostname, timeout = 10)
  output = `ssh -o ConnectTimeout=#{timeout} #{hostname} uptime`
  output.split(/\s+/)[-3..-1]
end

# notify that something is happening
STDERR.write "Checking hive{1..#{HIVE_COUNT}} uptime over SSH. Please hold.\n"

# get and sort server uptimes
uses = []

(1..HIVE_COUNT).each do |n|
  next if BANNED.include? n

  hive = "hive#{n}.cs.berkeley.edu"
  uses.push HostLoad.new(get_uptime(hive).map{|s| s.to_f}, hive)
end

puts uses.sort.map {|u| u.to_s}.join("\n")
