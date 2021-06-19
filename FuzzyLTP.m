% This code can be applied when N = 8 and radius = 1

function FuzzyLTP = FLTP(Proteins_PSSM,num)

for ii = 1:num
   img=Proteins_PSSM{1,ii};
    % img=imread('lena.jpg');
    % img=rgb2gray(img)
    imgSize=size(img);
    if numel(imgSize) > 2
        imgG = rgb2gray(img);
    else
        imgG = img;
    end

    imgG=double(imgG);
    [rows,cols]=size(imgG);
    
    imgn_upper=zeros(rows,cols);%Initialization result matrix (positive value)
    imgn_lower=zeros(rows,cols);%Initialization result matrix (negative value)
    for i=2:rows-1
       for j=2:cols-1 
            for p=i-1:i+1%Traversing peripheral pixels
                for q =j-1:j+1
                     %%% compute FLTP_upper
                     t=(imgG(p,q) - imgG(i,j))/imgG(i,j); 
                     % calculate dynamic threshold t
                     if (imgG(p,q) - imgG(i,j))>t
                         t=(imgG(p,q) - imgG(i,j))/imgG(i,j);
                         % reassign dynamic threshold t
                           
                         if(p==i-1&&q==j-1)                   
                             imgn_upper(i,j)=imgn_upper(i,j)+2^6;
                         end
                            
                         if(p==i-1&&q==j)                
                             imgn_upper(i,j)=imgn_upper(i,j)+2^7;
                         end
                            
                         if(p==i-1&&q==j+1)                 
                             imgn_upper(i,j)=imgn_upper(i,j)+2^0;
                         end
                            
                         if(p==i&&q==j-1)                    
                             imgn_upper(i,j)=imgn_upper(i,j)+2^5;
                         end
                            
                         if(p==i&&q==j+1)                    
                             imgn_upper(i,j)=imgn_upper(i,j)+2^1;
                         end
                            
                         if(p==i+1&&q==j-1)                    
                             imgn_upper(i,j)=imgn_upper(i,j)+2^4;
                         end
                            
                         if(p==i+1&&q==j)                    
                             imgn_upper(i,j)=imgn_upper(i,j)+2^3;
                         end
                            
                         if(p==i+1&&q==j+1)                    
                             imgn_upper(i,j)=imgn_upper(i,j)+2^2;
                         end
                          
                          %%% FLTP_lower
                     t=(imgG(p,q) - imgG(i,j))/imgG(i,j);     
                     elseif (imgG(p,q) - imgG(i,j))<-t||(imgG(p,q) - img(i,j))==-t
                         t=(imgG(p,q) - imgG(i,j))/imgG(i,j);
                         if(p==i-1&&q==j-1)                   
                             imgn_lower(i,j)=imgn_upper(i,j)+2^6;
                         end

                         if(p==i-1&&q==j)                
                             imgn_lower(i,j)=imgn_upper(i,j)+2^7;
                         end

                         if(p==i-1&&q==j+1)                 
                             imgn_lower(i,j)=imgn_upper(i,j)+2^0;
                         end

                         if(p==i&&q==j-1)                    
                             imgn_lower(i,j)=imgn_upper(i,j)+2^5;
                         end

                         if(p==i&&q==j+1)                    
                             imgn_lower(i,j)=imgn_upper(i,j)+2^1;
                         end

                         if(p==i+1&&q==j-1)                    
                             imgn_lower(i,j)=imgn_upper(i,j)+2^4;
                         end

                         if(p==i+1&&q==j)                    
                             imgn_lower(i,j)=imgn_upper(i,j)+2^3;
                         end

                         if(p==i+1&&q==j+1)                    
                             imgn_lower(i,j)=imgn_upper(i,j)+2^2;
                         end
                         
                    end
                end
            end
       end
    end
    % subplot(1,2,1),imshow(imgn_upper,[]),title('FLTP_upper');
    % subplot(1,2,2),imshow(imgn_lower,[]),title('FLTP_lower');
    new = uint8(imgn_lower);
    FLTP_lower = hist(new(:),0:255);
    FLTP_lower=FLTP_lower/sum(FLTP_lower);
    
    new1 = uint8(imgn_upper);
    FLTP_upper = hist(new1(:),0:255);
    FLTP_upper=FLTP_upper/sum(FLTP_upper);
    feature = [FLTP_lower,FLTP_upper];
    
    Feature_NB{ii} = feature; 
    FuzzyLTP = Feature_NB{ii};
end
