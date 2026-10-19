use strict;
use warnings;
no warnings "once";

my ($flesh, $peg, @files) = @ARGV;
die "usage: perl piraha_dump.pl FLESH PEG FILE...\n" unless defined $peg && @files;
unshift @INC, "$flesh/lib/sbin";
require Piraha;

my ($grammar, $rule) = piraha::parse_peg_file("$flesh/src/piraha/pegs/$peg");
my $reparse = defined $grammar->{patterns}->{stringparser};

sub mark_string_comments {
  my ($text) = @_;
  my $m = new Matcher($grammar, "stringparser", $text);
  return $text unless $m->matches();
  my $out = "";
  for (my $i = 0; $i < $m->groupCount(); $i++) {
    my $g = $m->group($i);
    my $s = $g->substring();
    $out .= $g->getPatternName() eq "stringcomment" ? "\x01$s\x02" : $s;
  }
  return $out;
}

sub reparse_strings {
  my ($node) = @_;
  if ($node->{name} eq "inquot") {
    my $marked = mark_string_comments($node->substring());
    @$node{qw(text start end children)} = ($marked, 0, length($marked), []);
    return;
  }
  reparse_strings($_) for @{$node->{children}};
}

binmode STDOUT;
for my $file (@files) {
  $main::indent = 0;
  my $matcher = piraha::parse_src($grammar, $rule, $file);
  if ($matcher->matches()) {
    reparse_strings($matcher->{gr}) if $reparse;
    print "OK\n", $matcher->{gr}->dump(), "\n\0";
  } else {
    print "ERROR\n", $matcher->showError(), "\0";
  }
}
