use strict;

my ($flesh, $peg, @files) = @ARGV;
die "usage: perl piraha_dump.pl FLESH PEG FILE...\n" unless defined $peg && @files;
unshift @INC, "$flesh/lib/sbin";
require Piraha;

my ($grammar, $rule) = piraha::parse_peg_file("$flesh/src/piraha/pegs/$peg");
for my $file (@files) {
  my $matcher = piraha::parse_src($grammar, $rule, $file);
  if ($matcher->matches()) {
    print "OK\n", $matcher->{gr}->dump(), "\n\0";
  } else {
    print "ERROR\n", $matcher->showError(), "\n\0";
  }
}
