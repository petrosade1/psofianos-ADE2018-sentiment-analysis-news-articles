<?php 
require("vendor/autoload.php");
 
$tagger = new \PosTagger();
$tagger->loadModelFromFile("bin/model_thre_0.49.bin");
fclose(STDIN);
fclose(STDOUT);
$STDIN = fopen('/dev/null', 'r');
$STDOUT = fopen('posout4.txt', 'wb');


 
$handle = fopen("output2.txt", "r");
if ($handle) {
    while (($line = fgets($handle)) !== false) {
        $tokens=explode(" ",$line);
        
        $tags = $tagger->tag($tokens);
        foreach ( $tokens as $index => $token){
        
        echo $token . " " . $tags[$index] ." ";
        
}


    }

    fclose($handle);
} else {
    
}

?>
