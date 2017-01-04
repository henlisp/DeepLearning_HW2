#!/usr/bin/perl -w ###-d:ptkdb

use strict;
use warnings;

#my $src = shift;
#my $dst = shift;
#if(!defined($src) || !defined($dst)) {
#    print "Error: <src> and <dst> files are not provided!\n";
#    exit;
#}
my $outputFileName  = "result_out.txt";
my @arrDropOut      = (0.5); # (0, 0.1, 0.5);
my @arrOptMethod    = ("adam");	# ("sgd", "adam", "adagrad");
my @arrEpochNum     = (100);# (20, 40, 60);
my @arrBatchNum     = (512);#(2, 32, 128);
my @arrNetwork      = ("n3");
my %bestResult      = (
        optimizationMethod  => "None",
        numberOfEpochs      => 0,
        batchNum            => 0,
        dropOut             => 0,
        result              => 0,
);

################################################################################
# Main
################################################################################
open(my $logbook, '>', "logbookFINALMODEL.log");
foreach my $opt (@arrOptMethod) {
    foreach my $epochNum (@arrEpochNum) {
        foreach my $batchNum (@arrBatchNum) {
            foreach my $dropOut (@arrDropOut) {
		foreach my $network (@arrNetwork) {
                    my $timeStart = time();
                    runTorch($opt, $epochNum, $batchNum, $dropOut, $network);
                    readResult($opt, $epochNum, $batchNum, $dropOut, $network, $timeStart);
                }
            }
        }
    }
}
printBestResult();
close($logbook);

################################################################################
# Functions
################################################################################

sub printMsg {
    my $msg = shift;
    print "$msg\n";
    print $logbook "$msg\n";
}

sub runTorch {
    my $opt         = shift;
    my $epochNum    = shift;
    my $batchNum    = shift;
    my $dropOut     = shift;
    my $network     = shift;

    #system("echo $opt $epochNum $batchNum");

    #my $res = rand(100);
    #print "Result: $res\n";
    
    #open(my $fh, '>', $outputFileName);
    #print $fh "$res\n";
    #close $fh;
	
    my $datestring = localtime();
    my $cmdLine = "th t5.FINALMODEL.lua -n $network -e $epochNum -o $opt -b $batchNum -f $outputFileName -d $dropOut";
    printMsg("========================================");
    printMsg("Time: $datestring");
    printMsg("$cmdLine");
    printMsg("========================================");
    system("$cmdLine");
}

sub readResult {
    my $opt         = shift;
    my $epochNum    = shift;
    my $batchNum    = shift;
    my $dropOut     = shift;
    my $network     = shift;
    my $timeStart   = shift;

    open (FILE, "$outputFileName");
    my @text=<FILE>;
    close FILE;
    
    my $res = $text[0];
    my $timeDelta = time() - $timeStart;
    printMsg("========================================");
    printMsg("Runtime:     $timeDelta seconds");
    printMsg("Parameters:  -o $opt -e $epochNum -b $batchNum -d $dropOut -n $network");
    printMsg("Result:      $res");
    printMsg("========================================");

    if($res > $bestResult{result}) {
        $bestResult{optimizationMethod} = $opt;
        $bestResult{numberOfEpochs}     = $epochNum;
        $bestResult{batchNum}           = $batchNum;
        $bestResult{dropOut}            = $dropOut;
        $bestResult{network}            = $network;
        $bestResult{result}             = $res;
     }
}

sub printBestResult {
    printMsg("BestResult: $bestResult{result}");
    printMsg("  optimizationMethod: $bestResult{optimizationMethod}");
    printMsg("  numberOfEpochs:     $bestResult{numberOfEpochs}");
    printMsg("  batchNum:           $bestResult{batchNum}");
    printMsg("  dropOut:            $bestResult{dropOut}");
    printMsg("  network:            $bestResult{network}");
}
