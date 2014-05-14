for ii=1:500
     imagesc(reshape(weights(:,ii), [78, 1901]),[-1 1]);
     title(num2str(ii));
     colorbar;
     input('');
end